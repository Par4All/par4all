# -*- coding: utf-8 -*-

from pyramid.view import view_config


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
                                        'tool_results',
                                        'tool_results_name',
                                        'compile',
                                        'perform',
                                        'perform_multiple',
                                        'dependence_graph',
                                        'dependence_graph_multi',
                                        ) ])
