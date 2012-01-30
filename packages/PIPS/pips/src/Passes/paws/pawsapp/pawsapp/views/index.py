# -*- coding: utf-8 -*-

import os, json

from pyramid.view import view_config


@view_config(route_name='home', renderer='pawsapp:templates/index.mako', http_cache=3600)
def index(request):
    """PAWS home page
    """
    valid_path = request.registry.settings['paws.validation']

    # Introductory text
    text = file(os.path.join(valid_path, 'main', 'paws.txt')).read()

    # Sections
    sections = json.load(file(os.path.join(valid_path, 'main', 'functionalities.txt')))
    for s in sections:
        path = os.path.join(valid_path, os.path.basename(s['path']))
        s['tools'] = [ { 'name'  : t,
                         'descr' : file(os.path.join(path, t, '%s.txt' % t)).read(),
                         }
                       for t in os.listdir(path) if not t.startswith('.') 
                       ]

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

    
    
