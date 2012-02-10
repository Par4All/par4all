# -*- coding: utf-8 -*-

import os

from pyramid.view import view_config


@view_config(route_name='home', renderer='pawsapp:templates/index.mako', http_cache=3600)
def index(request):
    """PAWS home page
    """
    valid_path = request.registry.settings['paws.validation']

    # Introductory text
    text = file(os.path.join(valid_path, 'main', 'paws.txt')).read()

    # Sections
    sections = request.site_sections

    return dict(text=text, sections = sections)
