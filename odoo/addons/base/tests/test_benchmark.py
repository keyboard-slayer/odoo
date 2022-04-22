# -*- coding: utf-8 -*-

import timeit
from inspect import cleandoc
from odoo.addons.base.models.ir_qweb import QWebException
from odoo.addons.mail.tests import common
from lxml import html


class SafeExprQWebTester(common.MailCommon):
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

    def test_benchmark(self):
        partner = self.env["res.partner"]

        test_partner_id = partner.create(
            {
                "name": "Johnny Test",
                "lang": "en_US",
                "comment": "A very good person, but a bit too experimental",
            }
        )

        code = cleandoc(
            """
        <p><t t-esc="object.name" /></p>
        <p><t t-esc="object.lang" /></p>
        <t t-esc="object.comment" />
        """
        )

        self.render_qweb_render_mixin(code, test_partner_id._name, test_partner_id.ids, bench=True)[
            test_partner_id.id
        ]
