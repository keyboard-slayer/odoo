# -*- coding: utf-8 -*-

import datetime
from os import system
from inspect import cleandoc
from collections import OrderedDict

from odoo.tests.common import BaseCase
from odoo.tools.safe_eval2 import safe_eval
from odoo.tools.expr_checker import expr_checker

class CustomCollection:
    def __init__(self, value):
        pass

    def __setitem__(self, key, value):
        pass

    def __getitem__(self, key):
        pass

    def __iter__(self):
        return iter([42])


class Dangerous:
    def say_goodbye(self):
        return "GoodBye"

    def __add__(self, val):
        print("Pwned !")


class Good:
    a = "hi"
    evil = Dangerous()
    eevil = 4

    _secret_stuff = "1+1=3"

    def tell_me_hi(self):
        return self.a

    def gift_of_satan(self):
        return 666

    def gather_secret(self):
        return self._secret_stuff

    def say_something_else(self, a):
        self.a = a


def check_type(method, value):
    safe_type = (Good, CustomCollection, OrderedDict)
    return type(value) in safe_type


class TestFuncChecker(BaseCase):
    def test_function_call(self):
        def abc(a, b, c):
            return f"A: {a}, B: {b}, C: {c}"

        code = cleandoc(
            """
            abc('aaa', 'bbb', 'ccc')
            """
        )

        safe_eval(code, locals_dict={"abc": abc})

        with self.assertRaisesRegex(
            ValueError, "safe_eval didn't like <.+.Dangerous object at .+>"
        ):
            code = cleandoc(
                """
                abc(a='aaa', c='ccc', b=Dangerous())
                """
            )

            safe_eval(code, locals_dict={"abc": abc, "Dangerous": Dangerous})

    def test_comp_expr(self):
        code = cleandoc(
            """
            [(lambda x: x**2)(n) for n in range(1, 11)]
            """
        )
        self.assertEqual(safe_eval(code), [1, 4, 9, 16, 25, 36, 49, 64, 81, 100])

    def test_attribute(self):
        a = Good()

        safe_eval("a.a", locals_dict={"a": a})
        safe_eval("a.tell_me_hi()", locals_dict={"a": a}, check_type=check_type)

        with self.assertRaisesRegex(
            ValueError, "safe_eval doesn't permit you to read eevil"
        ):
            safe_eval("a.eevil", locals_dict={"a": a})

        with self.assertRaisesRegex(
            ValueError, "safe_eval doesn't permit you to read gather_secret"
        ):
            safe_eval("a.gather_secret()", locals_dict={"a": a})

    def test_delete_attr(self):
        with self.assertRaisesRegex(
            ValueError, "safe_eval: doesn't permit you to delete attributes"
        ):
            safe_eval("del a.x")

    def test_method_return(self):
        code = cleandoc(
            """
            Good().tell_me_hi()
            """
        )
        safe_eval(code, locals_dict={"Good": Good}, check_type=check_type)

        with self.assertRaisesRegex(
            ValueError, "safe_eval doesn't permit you to read gift_of_satan"
        ):
            code = cleandoc(
                """
                Good().gift_of_satan()
                """
            )
            safe_eval(code, locals_dict={"Good": Good}, check_type=check_type)

    def test_function_return(self):
        def foo():
            return Good()

        def anti_foo():
            return Dangerous()

        safe_eval("foo()", locals_dict={"foo": foo}, check_type=check_type)

        with self.assertRaisesRegex(
            ValueError, "safe_eval didn't like <.+.Dangerous object at .+>"
        ):
            safe_eval(
                "anti_foo()", locals_dict={"anti_foo": anti_foo}, check_type=check_type
            )

    def test_evil_decorator(self):
        def evil(func):
            return lambda: Dangerous()

        @evil
        def foo():
            pass

        with self.assertRaisesRegex(
            ValueError, "safe_eval didn't like <.+.Dangerous object at .+>"
        ):
            safe_eval(
                "foo()", locals_dict={"foo": foo, "evil": evil}, check_type=check_type
            )

    def test_from_good_to_dangerous(self):
        a = Good()

        with self.assertRaisesRegex(
            ValueError, "safe_eval: doesn't permit you to store values in attributes"
        ):
            safe_eval(
                "a.__class__ = Dangerous",
                locals_dict={"Good": Good, "Dangerous": Dangerous},
                check_type=check_type,
            )

    def test_forbidden_attr(self):
        a = Good()

        with self.assertRaisesRegex(
            ValueError, "safe_eval doesn't permit you to read _secret_stuff"
        ):
            safe_eval("a._secret_stuff", locals_dict={"a": a})

        # FIXME with ast_set_attr
        with self.assertRaisesRegex(
            ValueError, "safe_eval: doesn't permit you to store values in attributes"
        ):
            safe_eval("a._secret_stuff = 42", mode="exec", locals_dict={"a": a})

    def test_deny_function_call(self):
        with self.assertRaises(Exception) as e:
            safe_eval("print('Hello, World')", allow_functions_calls=False)

        self.assertEqual(
            e.exception.args[0], "safe_eval didn't permit you to call any functions"
        )

        with self.assertRaisesRegex(
            Exception, "safe_eval didn't permit you to call any functions"
        ):
            safe_eval("kanban.get('sold')", allow_functions_calls=False)

    def test_assign(self):
        safe_eval("a = 4", mode="exec")
        safe_eval("a, b = 4, 5", mode="exec")

        with self.assertRaisesRegex(ValueError, "<.+.Dangerous object at .+>"):
            safe_eval(
                "a = Dangerous()", mode="exec", locals_dict={"Dangerous": Dangerous}
            )

        with self.assertRaisesRegex(ValueError, "<.+.Dangerous object at .+>"):
            safe_eval(
                "a, b = Good(), Dangerous()",
                mode="exec",
                locals_dict={"Dangerous": Dangerous, "Good": Good},
                check_type=check_type,
            )

    def test_dangerous_lambda(self):
        safe_eval("(lambda: 4)()", mode="exec")

        with self.assertRaisesRegex(ValueError, "<.+.Dangerous object at .+>"):
            safe_eval(
                "(lambda : print(Dangerous()))()",
                mode="exec",
                globals_dict={"Dangerous": Dangerous},
            )

    def test_overwrite(self):
        a = Good()

        with self.assertRaisesRegex(
            NameError, "safe_eval: __ast_check_type_fn is a reserved name"
        ):
            code = cleandoc(
                """
                def __ast_check_type_fn(t):
                    return t
                """
            )

            safe_eval(code, check_type=check_type, locals_dict={"a": a}, mode="exec")

        with self.assertRaisesRegex(
            NameError, "safe_eval: __ast_check_type_fn is a reserved name"
        ):
            code = cleandoc(
                """
                __ast_check_type_fn = lambda t: t
                """
            )

            safe_eval(code, check_type=check_type, locals_dict={"a": a}, mode="exec")

    def test_function_body(self):
        Good.test = Dangerous()
        code = cleandoc(
            """
               def b():
                   print(Good.test)
               b()
           """
        )

        with self.assertRaisesRegex(
            ValueError, "safe_eval doesn't permit you to read test"
        ):
            safe_eval(
                code, check_type=check_type, globals_dict={"Good": Good}, mode="exec"
            )

    def test_method_call(self):
        a = Good()
        d = Dangerous()

        safe_eval(
            "a.say_something_else('Hello')",
            mode="exec",
            locals_dict={"a": a},
            check_type=check_type,
        )

        with self.assertRaisesRegex(ValueError, "<.+.Dangerous object at .+>"):
            safe_eval(
                "d.say_goodbye()",
                mode="exec",
                locals_dict={"d": d},
                check_type=check_type,
            )

    def test_basics(self):
        codes = [
            (
                """
                result['val'] = 0
                for i in range(2, 10):
                    result['val'] += i
                """
            ),
            (
                """
                result['val'] = sum(range(5))
                """
            ),
            (
                """
                result['val'] = list(range(5))[-1]
                """
            ),
            (
                """
                result['val'] = list(range(5))[1:-1]
                """
            ),
            (
                """
                def gen():
                    for i in range(5):
                        yield 2 * 5
                
                result['val'] = sum(gen())
                """
            ),
            (
                """
                result['val'] = [x * 2 for x in range(5)]
                """
            ),
            (
                """
                a = 4
                b = 5
                c = a + b
                result['val'] = c // 3
                """
            ),
            (
                """
                a = 4
                b = 5
                c = a + b
                d = 0

                if c == 9:
                    d = c * 2
                else:
                    d = c

                result['val'] = d
                """
            ),
            (
                """
                result['val'] = "Hello" + ",World !"
                result['val'] += 6 * "PythonProgrammingIsFun"
                result['val'] = result['val'][::-1]
                """
            ),
            (
                """
                m = "Monkey"
                l = ["a", "b"]

                result['val'] = l.__str__()
                result['val'] += str(m.__len__())
                """
            ),
            (
                """
                i = 0
                result['val'] = ''

                while i < 10:
                    result['val'] += chr(65+i)
                    i += 1
                """
            ),
        ]

        for code in codes:
            output = {}
            expected_output = {}
            code = cleandoc(code)
            safe_eval(code, mode="exec", locals_dict={"result": output})
            exec(code, None, {"result": expected_output})

            self.assertEqual(output["val"], expected_output["val"])

    def test_safe_self(self):
        obj = Good()

        code = "Good.tell_me_hi(Good())"
        safe_eval(code, check_type=check_type, mode="eval", locals_dict={"Good": Good})

        code = "obj.tell_me_hi()"
        safe_eval(code, check_type=check_type, mode="eval", locals_dict={"obj": obj})

        code = cleandoc(
            """
            def foo(self):  
                return self

            ret['val'] = foo(65535)
            """
        )

        value = {}
        safe_eval(code, check_type=check_type, mode="exec", locals_dict={"ret": value})
        self.assertEqual(value["val"], 65535)

        # Because we use qualname to detect non-bounded classes (class.method)
        # It's safer to test those cases

        # Lambda functions inside of a listcomps are interpreted as <listcomp>.<lambda>
        code = cleandoc(
            """
            value['ret'] = [(lambda x: x**2)(n) for n in range(1, 11)] 
            """
        )

        value = {}
        safe_eval(
            code,
            check_type=check_type,
            mode="exec",
            globals_dict={"obj": obj, "value": value},
        )
        self.assertEqual(value["ret"], [1, 4, 9, 16, 25, 36, 49, 64, 81, 100])

        # Functions inside of functions are interpreted as a.<locals>.b
        code = cleandoc(
            """
            def a():
                def b():
                    pass
                b()

            a()
            """
        )
        safe_eval(code, mode="exec")

        code = cleandoc(
            """
            a = {'a': 'b', 'c': 'd'}
            a.get('a')
            """
        )
        safe_eval(code, mode="exec")

        with self.assertRaises(ValueError):
            code = "Good.tell_me_hi('hi')"
            safe_eval(
                code, check_type=check_type, mode="eval", locals_dict={"Good": Good}
            )

    def test_subscript(self):
        result = {}
        code = cleandoc(
            """ 
            result['list'] = [1, 2, 3, 4]
            result['list'][0] *= 2
            result['list'] = result['list'][::-1]
            """
        )

        safe_eval(code, mode="exec", locals_dict={"result": result})
        self.assertEqual(result["list"], [4, 3, 2, 2])

        code = cleandoc(
            """ 
            result['list2'] = [1, 2, 3, 4]
            result['list2'] = result['list2'][2:0:-1]
            """
        )

        safe_eval(code, mode="exec", locals_dict={"result": result})
        self.assertEqual(result["list2"], [3, 2])

        code = cleandoc(
            """
            result['dict'] = {}
            result['dict']['Hello'] = 'World !'
            result['dict'][45] = 99
            """
        )

        safe_eval(code, mode="exec", locals_dict={"result": result})
        self.assertEqual(result["dict"], {"Hello": "World !", 45: 99})

        code = cleandoc(
            """ 
            result['sum_all'] = 0

            result['all'] = [
                CustomCollection(44),
                [1, 2, 3, 4],
                {1: "Hello", 5: "World !", 9: "Foobar"},
                (5, 4, 3, 0),
                OrderedDict({2: "Ok", 4: "Nooo", 8: "Nice"}),
                {4, 9, 2, 22}
            ]

            for i in enumerate(result['all']):
                result['sum_all'] += sum(result['all'][i[0]])
            """
        )

        safe_eval(
            code,
            mode="exec",
            check_type=check_type,
            locals_dict={
                "result": result,
                "CustomCollection": CustomCollection,
                "OrderedDict": OrderedDict,
            },
        )

        expected_output = {}
        exec(
            code,
            None,
            {
                "result": expected_output,
                "CustomColletion": CustomCollection,
                "OrderedDict": OrderedDict,
            },
        )
        self.assertEqual(result["sum_all"], expected_output["sum_all"])

    def test_functional_functions(self):
        code = "list(map(lambda x: x*2, [1, 2, 3, 4]))"
        self.assertEqual(safe_eval(code), [2, 4, 6, 8])

        with self.assertRaisesRegex(ValueError, "<.+.Dangerous object at .+>"):
            code = "list(map(lambda x: Dangerous(), [1]))"
            safe_eval(code, mode="exec", globals_dict={"Dangerous": Dangerous})

    def test_sketchy_function_body(self):
        with self.assertRaisesRegex(ValueError, "<.+.Dangerous object at .+>"):
            code = cleandoc(
                """
                def evil_function(good_arg):
                    Dangerous()
                    return good_arg
                
                list(map(evil_function, [1, 2, 3]))
                """
            )

            safe_eval(code, mode="exec", globals_dict={"Dangerous": Dangerous})

    def test_sketchy_mapping(self):
        with self.assertRaisesRegex(ValueError, "<.+.Dangerous object at .+>"):
            code = cleandoc(
                """
                a = map(lambda x: Dangerous(), [1])
                for x in a:
                    x + 2
                """
            )

            safe_eval(code, mode="exec", globals_dict={"Dangerous": Dangerous})

    def test_common_bypass(self):
        with self.assertRaisesRegex(ValueError, "safe_eval doesn't permit you to read sys"):
            code = cleandoc(
                """
                datetime.sys.modules['os'].system('ls -la')
                """
            )

            safe_eval(code, globals_dict={"datetime": datetime})

        # FIXME !
        # with self.assertRaisesRegex(ValueError, "safe_eval didn't like <built-in function system>"):
        #     code = cleandoc(
        #         """ 
        #         system("ls -la")
        #         """
        #     )
        #     safe_eval(code, locals_dict={"system": system})
        
        code = cleandoc(
            """ 
            def myfunc():
                Dangerous()

            getattr(myfunc, '__globals__')['__ast_check_type'] = lambda x: x
            """ 
        )

        safe_eval(code, mode="exec", locals_dict={"Dangerous": Dangerous, "datetime": datetime})

