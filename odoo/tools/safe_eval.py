# -*- coding: utf-8 -*-
# Part of Odoo. See LICENSE file for full copyright and licensing details.

"""
safe_eval module - methods intended to provide more restricted alternatives to
                   evaluate simple and/or untrusted code.

Methods in this module are typically used as alternatives to eval() to parse
OpenERP domain strings, conditions and expressions, mostly based on locals
condition/math builtins.
"""

# Module partially ripped from/inspired by several different sources:
#  - http://code.activestate.com/recipes/286134/
#  - safe_eval in lp:~xrg/openobject-server/optimize-5.0
#  - safe_eval in tryton http://hg.tryton.org/hgwebdir.cgi/trytond/rev/bbb5f73319ad
import ast
import enum
import dis
import datetime
from dateutil.relativedelta import relativedelta
import functools
from inspect import cleandoc, getsource
import logging
import types
from opcode import HAVE_ARGUMENT, opmap, opname
from types import CodeType
import sys

import werkzeug
from psycopg2 import OperationalError

from .misc import ustr, frozendict

import odoo

unsafe_eval = eval

__all__ = ['test_expr', 'safe_eval', 'const_eval']

# The time module is usually already provided in the safe_eval environment
# but some code, e.g. datetime.datetime.now() (Windows/Python 2.5.2, bug
# lp:703841), does import time.
_ALLOWED_MODULES = ['_strptime', 'math', 'time']

_UNSAFE_ATTRIBUTES = ['f_builtins', 'f_globals', 'f_locals', 'gi_frame', 'gi_code',
                      'co_code', 'func_globals']

def to_opcodes(opnames, _opmap=opmap):
    for x in opnames:
        if x in _opmap:
            yield _opmap[x]
# opcodes which absolutely positively must not be usable in safe_eval,
# explicitly subtracted from all sets of valid opcodes just in case
_BLACKLIST = set(to_opcodes([
    # can't provide access to accessing arbitrary modules
    'IMPORT_STAR', 'IMPORT_NAME', 'IMPORT_FROM',
    # could allow replacing or updating core attributes on models & al, setitem
    # can be used to set field values
    'STORE_ATTR', 'DELETE_ATTR',
    # no reason to allow this
    'STORE_GLOBAL', 'DELETE_GLOBAL',
]))
# opcodes necessary to build literal values
_CONST_OPCODES = set(to_opcodes([
    # stack manipulations
    'POP_TOP', 'ROT_TWO', 'ROT_THREE', 'ROT_FOUR', 'DUP_TOP', 'DUP_TOP_TWO',
    'LOAD_CONST',
    'RETURN_VALUE', # return the result of the literal/expr evaluation
    # literal collections
    'BUILD_LIST', 'BUILD_MAP', 'BUILD_TUPLE', 'BUILD_SET',
    # 3.6: literal map with constant keys https://bugs.python.org/issue27140
    'BUILD_CONST_KEY_MAP',
    'LIST_EXTEND', 'SET_UPDATE',
])) - _BLACKLIST

# operations which are both binary and inplace, same order as in doc'
_operations = [
    'POWER', 'MULTIPLY', # 'MATRIX_MULTIPLY', # matrix operator (3.5+)
    'FLOOR_DIVIDE', 'TRUE_DIVIDE', 'MODULO', 'ADD',
    'SUBTRACT', 'LSHIFT', 'RSHIFT', 'AND', 'XOR', 'OR',
]
# operations on literal values
_EXPR_OPCODES = _CONST_OPCODES.union(to_opcodes([
    'UNARY_POSITIVE', 'UNARY_NEGATIVE', 'UNARY_NOT', 'UNARY_INVERT',
    *('BINARY_' + op for op in _operations), 'BINARY_SUBSCR',
    *('INPLACE_' + op for op in _operations),
    'BUILD_SLICE',
    # comprehensions
    'LIST_APPEND', 'MAP_ADD', 'SET_ADD',
    'COMPARE_OP',
    # specialised comparisons
    'IS_OP', 'CONTAINS_OP',
    'DICT_MERGE', 'DICT_UPDATE',
])) - _BLACKLIST

_SAFE_OPCODES = _EXPR_OPCODES.union(to_opcodes([
    'POP_BLOCK', 'POP_EXCEPT',

    # note: removed in 3.8
    'SETUP_LOOP', 'SETUP_EXCEPT', 'BREAK_LOOP', 'CONTINUE_LOOP',

    'EXTENDED_ARG',  # P3.6 for long jump offsets.
    'MAKE_FUNCTION', 'CALL_FUNCTION', 'CALL_FUNCTION_KW', 'CALL_FUNCTION_EX',
    # Added in P3.7 https://bugs.python.org/issue26110
    'CALL_METHOD', 'LOAD_METHOD',

    'GET_ITER', 'FOR_ITER', 'YIELD_VALUE',
    'JUMP_FORWARD', 'JUMP_ABSOLUTE',
    'JUMP_IF_FALSE_OR_POP', 'JUMP_IF_TRUE_OR_POP', 'POP_JUMP_IF_FALSE', 'POP_JUMP_IF_TRUE',
    'SETUP_FINALLY', 'END_FINALLY',
    # Added in 3.8 https://bugs.python.org/issue17611
    'BEGIN_FINALLY', 'CALL_FINALLY', 'POP_FINALLY',

    'RAISE_VARARGS', 'LOAD_NAME', 'STORE_NAME', 'DELETE_NAME', 'LOAD_ATTR',
    'LOAD_FAST', 'STORE_FAST', 'DELETE_FAST', 'UNPACK_SEQUENCE',
    'STORE_SUBSCR',
    'LOAD_GLOBAL',

    'RERAISE', 'JUMP_IF_NOT_EXC_MATCH',
])) - _BLACKLIST

_logger = logging.getLogger(__name__)

def assert_no_dunder_name(code_obj, expr):
    """ assert_no_dunder_name(code_obj, expr) -> None

    Asserts that the code object does not refer to any "dunder name"
    (__$name__), so that safe_eval prevents access to any internal-ish Python
    attribute or method (both are loaded via LOAD_ATTR which uses a name, not a
    const or a var).

    Checks that no such name exists in the provided code object (co_names).

    :param code_obj: code object to name-validate
    :type code_obj: CodeType
    :param str expr: expression corresponding to the code object, for debugging
                     purposes
    :raises NameError: in case a forbidden name (containing two underscores)
                       is found in ``code_obj``

    .. note:: actually forbids every name containing 2 underscores
    """
    for name in code_obj.co_names:
        if ("__" in name or name in _UNSAFE_ATTRIBUTES) and name not in (
            "__ast_check_fn", "__ast_check_type_fn", "__ast_check_attr_and_type"
        ):
            raise NameError('Access to forbidden name %r (%r)' % (name, expr))

def assert_valid_codeobj(allowed_codes, code_obj, expr):
    """ Asserts that the provided code object validates against the bytecode
    and name constraints.

    Recursively validates the code objects stored in its co_consts in case
    lambdas are being created/used (lambdas generate their own separated code
    objects and don't live in the root one)

    :param allowed_codes: list of permissible bytecode instructions
    :type allowed_codes: set(int)
    :param code_obj: code object to name-validate
    :type code_obj: CodeType
    :param str expr: expression corresponding to the code object, for debugging
                     purposes
    :raises ValueError: in case of forbidden bytecode in ``code_obj``
    :raises NameError: in case a forbidden name (containing two underscores)
                       is found in ``code_obj``
    """
    assert_no_dunder_name(code_obj, expr)

    # set operations are almost twice as fast as a manual iteration + condition
    # when loading /web according to line_profiler
    code_codes = {i.opcode for i in dis.get_instructions(code_obj)}
    if not allowed_codes >= code_codes:
        raise ValueError("forbidden opcode(s) in %r: %s" % (expr, ', '.join(opname[x] for x in (code_codes - allowed_codes))))

    for const in code_obj.co_consts:
        if isinstance(const, CodeType):
            # assert_valid_codeobj(allowed_codes, const, 'lambda')
            assert_valid_codeobj(allowed_codes, const, expr)

def test_expr(expr, allowed_codes, mode="eval"):
    """test_expr(expression, allowed_codes[, mode]) -> code_object

    Test that the expression contains only the allowed opcodes.
    If the expression is valid and contains only allowed codes,
    return the compiled code object.
    Otherwise raise a ValueError, a Syntax Error or TypeError accordingly.
    """
    try:
        if mode == 'eval':
            # eval() does not like leading/trailing whitespace
            expr = expr.strip()
        code_obj = compile(expr, "", mode)
    except (SyntaxError, TypeError, ValueError):
        raise
    except Exception as e:
        raise ValueError('"%s" while compiling\n%r' % (ustr(e), expr))
    assert_valid_codeobj(allowed_codes, code_obj, expr)
    return code_obj


def const_eval(expr):
    """const_eval(expression) -> value

    Safe Python constant evaluation

    Evaluates a string that contains an expression describing
    a Python constant. Strings that are not valid Python expressions
    or that contain other code besides the constant raise ValueError.

    >>> const_eval("10")
    10
    >>> const_eval("[1,2, (3,4), {'foo':'bar'}]")
    [1, 2, (3, 4), {'foo': 'bar'}]
    >>> const_eval("1+2")
    Traceback (most recent call last):
    ...
    ValueError: opcode BINARY_ADD not allowed
    """
    c = test_expr(expr, _CONST_OPCODES)
    return unsafe_eval(c)

def expr_eval(expr):
    """expr_eval(expression) -> value

    Restricted Python expression evaluation

    Evaluates a string that contains an expression that only
    uses Python constants. This can be used to e.g. evaluate
    a numerical expression from an untrusted source.

    >>> expr_eval("1+2")
    3
    >>> expr_eval("[1,2]*2")
    [1, 2, 1, 2]
    >>> expr_eval("__import__('sys').modules")
    Traceback (most recent call last):
    ...
    ValueError: opcode LOAD_NAME not allowed
    """
    c = test_expr(expr, _EXPR_OPCODES)
    return unsafe_eval(c)

def _import(name, globals=None, locals=None, fromlist=None, level=-1):
    if globals is None:
        globals = {}
    if locals is None:
        locals = {}
    if fromlist is None:
        fromlist = []
    if name in _ALLOWED_MODULES:
        return __import__(name, globals, locals, level)
    raise ImportError(name)
_BUILTINS = {
    '__import__': _import,
    'True': True,
    'False': False,
    'None': None,
    'bytes': bytes,
    'str': str,
    'unicode': str,
    'bool': bool,
    'int': int,
    'float': float,
    'enumerate': enumerate,
    'dict': dict,
    'list': list,
    'tuple': tuple,
    'map': map,
    'abs': abs,
    'min': min,
    'max': max,
    'sum': sum,
    'reduce': functools.reduce,
    'filter': filter,
    'sorted': sorted,
    'round': round,
    'len': len,
    'repr': repr,
    'set': set,
    'all': all,
    'any': any,
    'ord': ord,
    'chr': chr,
    'divmod': divmod,
    'isinstance': isinstance,
    'range': range,
    'xrange': range,
    'zip': zip,
    'Exception': Exception,
}

class SubscriptWrapper:
    """
    SubscriptWrapper

    :param value: The value of the subscript (can be any object that supports __getitem__ and __setitem__)
    :param check_type: A function that will check the arguments passed to the methods and also the return values
    """

    def __init__(self, value, check_type):
        self.value = value
        self.check_type = check_type

    def __getitem__(self, key):
        self.check_type("arguments", key)
        return self.check_type("returned", self.value.__getitem__(key))

    def __setitem__(self, key, value):
        self.check_type("arguments", key)
        self.check_type("constant", value)

        self.value.__setitem__(key, value)


class FuncWrapper:
    def __init__(self, function, check_fn, check_attr):
        self.function = function
        self.check_fn = check_fn
        self.check_attr = check_attr

    def __call__(self, *args, **kwargs):
        self.check_fn(self.function, *args, **kwargs)

    def __getattr__(self, name):
        return self.check_attr(self.function, name, getattr(self.function, name))

class wrap_module:
    def __init__(self, module, attributes):
        """Helper for wrapping a package/module to expose selected attributes

        :param module: the actual package/module to wrap, as returned by ``import <module>``
        :param iterable attributes: attributes to expose / whitelist. If a dict,
                                    the keys are the attributes and the values
                                    are used as an ``attributes`` in case the
                                    corresponding item is a submodule
        """
        # builtin modules don't have a __file__ at all
        modfile = getattr(module, '__file__', '(built-in)')
        self._repr = f"<wrapped {module.__name__!r} ({modfile})>"
        for attrib in attributes:
            target = getattr(module, attrib)
            if isinstance(target, types.ModuleType):
                target = wrap_module(target, attributes[attrib])
            setattr(self, attrib, target)

    def __repr__(self):
        return self._repr


__require_checks_type = (
    types.FunctionType,
    types.LambdaType,
    types.MethodType,
    types.BuiltinFunctionType,
    types.BuiltinMethodType,
    types.WrapperDescriptorType,
    types.MethodWrapperType,
    types.MethodDescriptorType,
    types.ClassMethodDescriptorType,
    types.ModuleType,
)

__safe_type = (
    str,
    bytes,
    float,
    complex,
    int,
    bool,
    type(None),
    tuple,
    list,
    set,
    dict,
    slice,
    enumerate,
    range,
    types.GeneratorType,
    map,
    FuncWrapper,
    SubscriptWrapper,
    wrap_module,
    datetime.date,
    datetime.datetime,
    datetime.timedelta,
    enum.EnumMeta,
    frozendict,
    odoo.api.Environment,
    relativedelta
)

class NodeChecker(ast.NodeTransformer):
    """
    https://docs.python.org/3/library/ast.html#ast.NodeTransformer
    """

    def __init__(self, allow_function_calls, allow_private):
        self.allow_function_calls = allow_function_calls
        self.allow_private = allow_private
        self.reserved_name = (
            "__ast_check_fn",
            "__ast_check_type_fn",
            "__ast_check_attr_and_type",
            "FuncWrapper",
            "SubscriptWrapper"
        )
        super().__init__()

    def visit_Call(self, node):
        node = self.generic_visit(node)

        if not self.allow_function_calls:
            raise Exception("safe_eval didn't permit you to call any functions")

        return ast.Call(
            func=ast.Name("__ast_check_fn", ctx=ast.Load()),
            args=[node.func] + node.args,
            keywords=node.keywords,
        )

    def visit_Name(self, node):
        node = self.generic_visit(node)

        if node.id in self.reserved_name:
            raise NameError(f"safe_eval: {node.id} is a reserved name")

        return node

    def visit_Subscript(self, node):
        node = self.generic_visit(node)

        node.value = ast.Call(
            ast.Name("SubscriptWrapper", ctx=ast.Load),
            args=[node.value, ast.Name("__ast_check_type_fn", ctx=ast.Load())],
            keywords={},
        )

        return node

    def visit_FunctionDef(self, node):
        node = self.generic_visit(node)

        if node.name in self.reserved_name:
            raise NameError(f"safe_eval: {node.name} is a reserved name")

        return node

    def visit_Attribute(self, node):
        node = self.generic_visit(node)

        if (
            isinstance(node.value, ast.Name)
            and node.value.id.startswith("_")
            and not self.priv
        ):
            raise NameError(f"safe_eval: didn't permit you to read private elements")

        if isinstance(node.ctx, ast.Load):
            return ast.Call(
                func=ast.Name("__ast_check_attr_and_type", ctx=ast.Load()),
                args=[node.value, ast.Constant(node.attr), node],
                keywords=[],
            )

        elif isinstance(node.ctx, ast.Store):
            raise ValueError(
                "safe_eval: doesn't permit you to store values in attributes"
            )

        elif isinstance(node.ctx, ast.Del):
            raise ValueError("safe_eval: doesn't permit you to delete attributes")


def is_unbound_method_call(func):
    """
    is_unbound_method_call(func) -> bool

    Checks that a function is an unbound method or not. Unlike bound method, it's complicated to determine
    if it's a function or a method. The only way we found was to check for the qualname (we intend class.method).
    The class and the method has to be a valid Python identifier and should have only one dot.

    :param func: A function type object
    :return bool: Is an unbound method or not
    """
    try:
        classname, methodname = func.__qualname__.split(".")
    except ValueError:
        # Probably a namespace (like TestFuncChecker.test_function_call.abc)
        return False

    if not classname.isidentifier() or not methodname.isidentifier():
        return False  # Probably smth like <listcomp>.<lambda>

    if type(func) == types.BuiltinMethodType or type(func) == types.MethodType:
        return False  # A bound method

    return True


def expr_checker_prepare_context(
    check_attr=None, return_code=False, check_type=None, check_function=None
):
    """
    expr_checker_prepare_context(check_attr, return_code, check_type, check_type) -> (dict | str)

    This function prepare the execution context for the sandbox, you should pass the execution context to an eval / exec function

    :param check_attr: A function that will check the methods / attributes according to their object, name and value and will return a boolean
                     eg: def __safe_check_attr(obj: object, name: str, value: Any) -> bool

    :param return_code: if True it will return a string with all the code needed for the execution, if False it will return a dictionnary {function_name: function_object}

    :param check_type: if not None will pass the check of type to a custom function (please refer to the docstring for __ast_default_check_type)
                        eg: def __safe_check_type(method: str, value: object) -> bool

    :param check_function: if not None will pass the function check to a custom function (please refer to the docstring for __ast_default_check_call)
                            eg: def __safe_check_function_call(func: FunctionType, *args, **kwargs) -> bool
    """

    def __ast_default_check_type(method, value):
        """
        __ast_default_check_type(method, value) -> value

        Also represented as check_type

        check the type of `value` against a whitelist and return the value.
        will the check if the plug-in function `check_type` (argument from the `prepare_context` function) is present or not.

        The plug-in function will take the same arguments as this function and should return a boolean (True if allowed and False otherwise)

        :param method: A string that represent the way the value interact with code this can be:
                        * returned: if it's returned from a function (eg: return value)
                        * arguments: if it's a function argument (eg: foo(value))
                        * constant: if it's a constant (eg: value)
                        * called: if it's a function call, it can be useful in case of bound method (with the __self__ attribute) (eg: value()).
                        * subscript: if it's the return value of a subscript
                        * self: if it's a method, represent the bound object

        :param value: A value that needs to be checked.
        :return: The value passed to this function.
        """

        if callable(value) and method in ["returned", "arguments", "subscript", "self"]:
            return FuncWrapper(value, __ast_default_check_call, __ast_check_attr_and_type)

        if type(value) == type and value in __safe_type:
            return value
        
        if not isinstance(value, odoo.models.BaseModel) and (
            type(value) not in __safe_type + __require_checks_type or (
                type(value) in __require_checks_type
                and method in ["returned", "arguments", "subscript", "self"])
            ):
            if not (check_type is not None and check_type(method, value)):
                raise ValueError(f"safe_eval didn't like {value} (type: {type(value)})")

        return value

    def __ast_default_check_call(func, *args, **kwargs):
        """
        __ast_default_check_call(func, check_type, *args, **kwargs) -> value

        Check functions / method calls.
        In case of a simple function / lambda, it will send its arguments and its return values to check_type.
        In case of an unbound method (e.g Class.method) it will ensure that the self origins from the same class as the method.
        In case of a bound method, it will ensure that the __self__ is safe.

        :param func: The called function
        :param args: Arguments to the given function
        :param kargs: Keywords arguments to the given function
        :return: The checked value from the called function
        """

        if func is None:
            return None

        if check_function is not None and check_function(func, *args, **kwargs):
            return __ast_default_check_type("returned", func(*args, **kwargs))

        if (
            func.__name__ == "get"
            and hasattr(func, "__self__")
            and type(func.__self__) == dict
        ):
            # If it's a dictionnary, we check types like it's a constant
            return __ast_default_check_type("constant", func(*args, **kwargs))

        for arg in (*args, *kwargs.values()):
            __ast_default_check_type("arguments", arg)

            if "." in func.__qualname__:
                if (
                    args
                    and (
                        is_unbound_method_call(func)
                        and not hasattr(args[0], func.__name__)
                    )
                    or (
                        args and 
                        hasattr(args[0], func.__name__)
                        and getattr(args[0], func.__name__).__func__ != func
                    )
                ):
                    raise ValueError(
                        "safe_eval didn't like method call without appropriate type"
                    )

        if hasattr(func, "__self__") and type(func) not in [types.BuiltinFunctionType, types.BuiltinMethodType]:
            __ast_default_check_type("self", func.__self__)

        return __ast_default_check_type("returned", func(*args, **kwargs))

    def __ast_check_attr(obj, key):
        """
        __ast_check_attr -> value

        Will check if the user is allowed to read a certain attribute from an object

        :param obj: The object with the attribute we want to read
        :param key: The attribute we want to read, represented as a string 
        """

        if not obj:
            return obj

        if type(obj) == odoo.api.Environment and key in ["env"] or \
            check_attr != None and check_attr(obj, key):
            return getattr(obj, key)
        
        return getattr(obj, key) # FIXME
        # raise ValueError(f"safe-eval didn't permit you to read {key} from {obj} (of type {type(obj)})")

    def __ast_check_attr_and_type(value, attr, node):
        """
        __ast_check_attr_and_type -> value

        A simple wrapper for check_type and check_attr (given by the user in the super function)


        :param value: an object we want to test
        :param attr: a string that respresent the attribute / method we want to test
        :param node: the value we want to test

        :return: the tested value
        """

        ret = __ast_default_check_type("attribute", __ast_check_attr(value, attr))

        if ret or (not ret and type(ret) == bool):
            return node
        
        return node #FIXME
        #raise ValueError(f"safe_eval doesn't permit you to read {attr} from {node} of type {type(node)}")

    if not return_code:
        return {
            "__ast_check_type_fn": __ast_default_check_type,
            "__ast_check_fn": __ast_default_check_call,
            "__ast_check_attr": __ast_check_attr,
            "__ast_check_attr_and_type": __ast_check_attr_and_type,
            "SubscriptWrapper": SubscriptWrapper,
            "FuncWrapper": FuncWrapper
        }

    else:
        pass
        # TODO
        # return "\n".join(
        #     [
        #         cleandoc(
        #             getsource(check_attr).replace(check_attr.__name__, "__ast_check_attr")
        #         ),
        #         cleandoc(
        #             getsource(check_type).replace(
        #                 check_type.__name__, "__ast_check_type_fn"
        #             )
        #         ),
        #         cleandoc(
        #             getsource(check_function).replace(
        #                 check_function.__name__, "__ast_check_fn"
        #             )
        #         ),
        #         user_code,
        #     ]
        # )


def expr_checker(
    expr,
    allow_function_calls=True,
    allow_private=False,
):
    """
    expr_checker(expr, allow_function_calls, allow_private, return_code) -> str

    Take a Python expression and will return an expression with different checks (look above ;-) )

    :param expr: A str that contains the expression to check
    :param allow_function_calls: If False will raise an exception when it will meet a function call
    :param allow_private: If True will raise an exception when it will meet a dunder attribute / method
    """
    node_checker = NodeChecker(allow_function_calls, allow_private)
    return ast.unparse(node_checker.visit(ast.parse(expr)))


def safe_eval(expr, globals_dict={}, allow_function_calls=True, allow_private=False, check_attr=None, check_type=None, check_function=None, *args, **kwargs):
    if sys.version_info.minor < 9:
        _logger.warning("Your version of Python is deprecated, please upgrade to 3.9 or later")
        _safe_eval_legacy(expr, globals_dict, *args, **kwargs)

    if "mode" not in kwargs or kwargs["mode"] == "eval":
        expr = expr.strip()

    checked_expr = expr_checker(expr, allow_function_calls=allow_function_calls, allow_private=allow_private)

    globals_dict.update(expr_checker_prepare_context(check_attr=check_attr, check_type=check_type, check_function=check_function))
    print(checked_expr)

    try:
        return _safe_eval_legacy(checked_expr, globals_dict, *args, **kwargs)
    except ValueError as e:
        raise ValueError('%s: "%s" while evaluating\n%r' % (ustr(type(e)), ustr(e), expr))


def _safe_eval_legacy(expr, globals_dict=None, locals_dict=None, mode="eval", nocopy=False, locals_builtins=False):
    """_safe_eval_legacy(expression[, globals[, locals[, mode[, nocopy]]]]) -> result

    System-restricted Python expression evaluation

    Evaluates a string that contains an expression that mostly
    uses Python constants, arithmetic expressions and the
    objects directly provided in context.

    This can be used to e.g. evaluate
    an OpenERP domain expression from an untrusted source.

    :throws TypeError: If the expression provided is a code object
    :throws SyntaxError: If the expression provided is not valid Python
    :throws NameError: If the expression provided accesses forbidden names
    :throws ValueError: If the expression provided uses forbidden bytecode
    """
    if type(expr) is CodeType:
        raise TypeError("safe_eval does not allow direct evaluation of code objects.")

    # prevent altering the globals/locals from within the sandbox
    # by taking a copy.
    if not nocopy:
        # isinstance() does not work below, we want *exactly* the dict class
        if (globals_dict is not None and type(globals_dict) is not dict) \
                or (locals_dict is not None and type(locals_dict) is not dict):
            _logger.warning(
                "Looks like you are trying to pass a dynamic environment, "
                "you should probably pass nocopy=True to safe_eval().")
        if globals_dict is not None:
            globals_dict = dict(globals_dict)
        if locals_dict is not None:
            locals_dict = dict(locals_dict)

    check_values(globals_dict)
    check_values(locals_dict)

    if globals_dict is None:
        globals_dict = {}

    globals_dict['__builtins__'] = _BUILTINS
    if locals_builtins:
        if locals_dict is None:
            locals_dict = {}
        locals_dict.update(_BUILTINS)
    c = test_expr(expr, _SAFE_OPCODES, mode=mode)
    try:
        return unsafe_eval(c, globals_dict, locals_dict)
    except odoo.exceptions.UserError:
        raise
    except odoo.exceptions.RedirectWarning:
        raise
    except werkzeug.exceptions.HTTPException:
        raise
    except OperationalError:
        # Do not hide PostgreSQL low-level exceptions, to let the auto-replay
        # of serialized transactions work its magic
        raise
    except ZeroDivisionError:
        raise
    except Exception as e:
        raise ValueError('%s: "%s" while evaluating\n%r' % (ustr(type(e)), ustr(e), expr))
def test_python_expr(expr, mode="eval"):
    try:
        test_expr(expr, _SAFE_OPCODES, mode=mode)
    except (SyntaxError, TypeError, ValueError) as err:
        if len(err.args) >= 2 and len(err.args[1]) >= 4:
            error = {
                'message': err.args[0],
                'filename': err.args[1][0],
                'lineno': err.args[1][1],
                'offset': err.args[1][2],
                'error_line': err.args[1][3],
            }
            msg = "%s : %s at line %d\n%s" % (type(err).__name__, error['message'], error['lineno'], error['error_line'])
        else:
            msg = ustr(err)
        return msg
    return False


def check_values(d):
    if not d:
        return d
    for v in d.values():
        if isinstance(v, types.ModuleType):
            raise TypeError(f"""Module {v} can not be used in evaluation contexts

Prefer providing only the items necessary for your intended use.

If a "module" is necessary for backwards compatibility, use
`odoo.tools.safe_eval.wrap_module` to generate a wrapper recursively
whitelisting allowed attributes.

Pre-wrapped modules are provided as attributes of `odoo.tools.safe_eval`.
""")
    return d

# dateutil submodules are lazy so need to import them for them to "exist"
import dateutil
mods = ['parser', 'relativedelta', 'rrule', 'tz']
for mod in mods:
    __import__('dateutil.%s' % mod)
datetime = wrap_module(__import__('datetime'), ['date', 'datetime', 'time', 'timedelta', 'timezone', 'tzinfo', 'MAXYEAR', 'MINYEAR'])
dateutil = wrap_module(dateutil, {
    mod: getattr(dateutil, mod).__all__
    for mod in mods
})
json = wrap_module(__import__('json'), ['loads', 'dumps'])
time = wrap_module(__import__('time'), ['time', 'strptime', 'strftime'])
pytz = wrap_module(__import__('pytz'), [
    'utc', 'UTC', 'timezone',
])
