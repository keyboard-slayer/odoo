# -*- coding: utf-8 -*-

import timeit
from inspect import cleandoc
from odoo.addons.base.models.ir_qweb import QWebException
from odoo.addons.mail.tests import common
from lxml import html


class SafeExprQWebTester(common.MailCommon):
    @classmethod
    def setUpClass(cls):
        super(SafeExprQWebTester, cls).setUpClass()
        _partner = cls.env["res.partner"]

        cls.test_partner_id = _partner.create(
            {
                "name": "Johnny Test",
                "lang": "en_US",
                "comment": "A very good person, but a bit too experimental",
            }
        )

    def render_qweb(self, expr, ctx={}, bench=False):
        if bench:
            with_checks = 0
            without_checks = 0

            for _ in range(5):
                with_checks += timeit.timeit(
                    cleandoc(
                        """
                    self.env['ir.qweb']._render(
                        html.fragment_fromstring(expr, create_parent="div"), ctx
                    )
                    """
                    ),
                    number=500,
                    globals={"self": self, "expr": expr, "ctx": ctx, "html": html},
                )

                without_checks += timeit.timeit(
                    cleandoc(
                        """
                    self.env['ir.qweb'].with_context(benchmark_mode=True)._render(
                        html.fragment_fromstring(expr, create_parent="div"), ctx
                    )
                    """
                    ),
                    number=500,
                    globals={"self": self, "expr": expr, "ctx": ctx, "html": html},
                )

            print(f"Average time without checks: {without_checks / 5}")
            print(f"Average time with checks: {with_checks / 5}")

        return self.env["ir.qweb"]._render(
            html.fragment_fromstring(expr, create_parent="div"), ctx
        )

    def render_qweb_render_mixin(self, expr, model, res_id, bench=False):
        with_checks = 0
        without_checks = 0

        if bench:
            for _ in range(5):
                with_checks += timeit.timeit(
                    cleandoc(
                        """
                    self.env["mail.render.mixin"]._render_template(
                        expr, model, res_id, engine="qweb"
                    )
                    """
                    ),
                    number=500,
                    globals={
                        "self": self,
                        "expr": expr,
                        "model": model,
                        "res_id": res_id,
                    },
                )

                without_checks += timeit.timeit(
                    cleandoc(
                        """ 
                    self.env["mail.render.mixin"].with_context(
                        benchmark_mode=True
                    )._render_template(expr, model, res_id, engine="qweb")
                    """
                    ),
                    number=500,
                    globals={
                        "self": self,
                        "expr": expr,
                        "model": model,
                        "res_id": res_id,
                    },
                )

            print(f"Average time without checks: {without_checks / 5}")
            print(f"Average time with checks: {with_checks / 5}")

        return self.env["mail.render.mixin"]._render_template(
            expr, model, res_id, engine="qweb"
        )

    def test_reflect_env(self):
        with self.assertRaises(ValueError):
            code = cleandoc(
                """ 
                <p t t-esc="object.env" /></p>
                """
            )

            self.render_qweb_render_mixin(code, self.test_partner_id._name, self.test_partner_id.ids)[
                self.test_partner_id.id
            ]

    def test_function_calls(self):
        with self.assertRaisesRegex(Exception, "qweb didn't permit you to call any functions"):
            code = cleandoc(
                """ 
                <t t-foreach="range(2, 20)" t-as="i">
                    <p><t t-esc='i' /></p>
                </t>
                """
            )

            self.render_qweb(code)


    def test_benchmark(self):
        code = cleandoc(
            """
        <p><t t-esc="object.name" /></p>
        <p><t t-esc="object.lang" /></p>
        <t t-esc="object.comment" />
        """
        )

        self.render_qweb_render_mixin(code, self.test_partner_id._name, self.test_partner_id.ids, bench=True)[
            self.test_partner_id.id
        ]

