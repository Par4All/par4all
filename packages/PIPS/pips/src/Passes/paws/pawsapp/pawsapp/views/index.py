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


@view_config(route_name='routes.js', renderer='pawsapp:templates/routes.js.mako', permission='view', http_cache=3600)
def routes(request):
    """Export selected routes to Javascript
    """
    introspector = request.registry.introspector

    request.response.content_type = 'text/javascript'
    return dict(routes = [ dict(name=name, pattern=introspector.get('routes', name)['pattern'])
                           for name in ('load_example_file',
                                        'source_panel',
                                        'get_directory',
                                        'get_functions',
                                        'detect_language',
                                        'compile',
                                        'perform',
                                        'perform_multiple',
                                        'dependence_graph',
                                        'dependence_graph_multi',
                                        ) ])

    
    
