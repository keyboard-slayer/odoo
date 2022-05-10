import ast
import traceback
from odoo.tools.expr_checker import expr_checker, expr_checker_prepare_context

unsafe_eval = eval

_BUILTINS = {
    "True": True,
    "False": False,
    "None": None,
    "bytes": bytes,
    "str": str,
    "unicode": str,
    "bool": bool,
    "int": int,
    "float": float,
    "enumerate": enumerate,
    "dict": dict,
    "list": list,
    "tuple": tuple,
    "map": map,
    "abs": abs,
    "min": min,
    "max": max,
    "sum": sum,
    "filter": filter,
    "sorted": sorted,
    "round": round,
    "len": len,
    "repr": repr,
    "set": set,
    "all": all,
    "any": any,
    "ord": ord,
    "chr": chr,
    "divmod": divmod,
    "isinstance": isinstance,
    "range": range,
    "xrange": range,
    "zip": zip,
    "Exception": Exception,
    "print": print,
    "type": type,
}


def safe_get_attr(obj, key):
    # NOTE: Those keys are for testing purpose
    return key in (
        "a",
        "get",
        "x",
        "tell_me_hi",
        "set_x_value",
        "say_something_else",
        "say_goodbye",
        "__str__",
        "__len__",
    )


def safe_eval(
    expr,
    globals_dict={},
    locals_dict={},
    mode="eval",
    check_type=None,
    allow_functions_calls=True,
    dbg=False,
):
    if globals_dict is not None:
        globals_dict = dict(globals_dict)

    if locals_dict is not None:
        locals_dict = dict(locals_dict)

    globals_dict["__builtins__"] = _BUILTINS

    if check_type is None:
        scope = expr_checker_prepare_context(safe_get_attr)
    else:
        scope = expr_checker_prepare_context(safe_get_attr, check_type=check_type)

    code = expr_checker(
        expr, allow_function_calls=allow_functions_calls, allow_private=True
    )
    globals_dict.update(scope)

    if dbg:
        print(" === ")
        print(expr)
        print(" === ")
        print(ast.dump(ast.parse(expr), indent=4))
        print(" === \n\n")
        print(code)
        input()

    c = compile(code, "", mode)

    try:
        return unsafe_eval(c, globals_dict, locals_dict)
    except Exception as e:
        if dbg:
            traceback.print_exc()
            input(e)
        else:
            raise e
