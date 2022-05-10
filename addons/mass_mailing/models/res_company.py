# -*- coding: utf-8 -*-
# Part of Odoo. See LICENSE file for full copyright and licensing details.

from odoo import fields, models


class ResCompany(models.Model):
    _inherit = "res.company"

    social_facebook_link = fields.Char("Facebook social media link", compute="_compute_social_media_links")
    social_linkedin_link = fields.Char("Linkedin social media link", compute="_compute_social_media_links")
    social_twitter_link = fields.Char("Twitter social media link", compute="_compute_social_media_links")
    social_instagram_link = fields.Char("Instagram social media link", compute="_compute_social_media_links")

    def _compute_social_media_links(self):
        for record in self:
            record.social_facebook_link = record.social_facebook
            record.social_linkedin_link = record.social_linkedin
            record.social_twitter_link = record.social_twitter
            record.social_instagram_link = record.social_instagram