# -*- coding: utf-8 -*-

"""
Generic tool controller

"""

import os

from pyramid.view import view_config


toolName = dict(
    preconditions = u'Preconditions over scalar integer variables',
    openmp        = u'Openmp demo page',
    in_regions    = u'IN regions',
    out_regions   = u'OUT regions',
    regions       = u'Array regions',
    )


def _get_examples(tool, request):
    """Get list of examples for the specified tool.
    """
    path = os.path.join(request.registry.settings['paws.validation'], 'tools', tool)
    return [ f for f in sorted(os.listdir(path))
             if os.path.isdir(f) == False
             and (f.endswith('.c') or f.endswith('.f'))]


@view_config(route_name='tool_basic',    renderer='pawsapp:templates/tool.mako')
@view_config(route_name='tool_advanced', renderer='pawsapp:templates/tool.mako')
def tool(request):
    """Generic tool view.
    """
    tool  = request.matchdict['tool']
    return dict(tool     = tool,
                descr    = toolName[tool],
                advanced = bool(request.matched_route.name.endswith('advanced')),
                examples = _get_examples(tool, request),
                )
