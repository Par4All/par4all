# -*- coding: utf-8 -*-

"""
Generic tool controller

"""
import os

from zipfile      import ZipFile
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
    path = os.path.join(request.registry.settings['paws.validation'], 'tools', os.path.basename(tool))
    return [ f for f in sorted(os.listdir(path))
             if os.path.isdir(f) == False
             and (f.endswith('.c') or f.endswith('.f'))]


@view_config(route_name='tool_basic',    renderer='pawsapp:templates/tool.mako')
@view_config(route_name='tool_advanced', renderer='pawsapp:templates/tool.mako')
def tool(request):
    """Generic tool view (basic and advanced modes).
    """
    tool  = request.matchdict['tool']
    return dict(tool     = tool,
                descr    = toolName[tool],
                advanced = bool(request.matched_route.name.endswith('advanced')),
                examples = _get_examples(tool, request),
                )


@view_config(route_name='upload_userfile', renderer='json')
def upload_userfile(request):
    """Handle file upload. Multiple files can be sent inside a single zip file.
    """
    source   = request.POST['file']
    filename = source.filename

    if os.path.splitext(filename)[1].lower() == '.zip':
        # Multiples source files inside a Zip file
        zip   = ZipFile(source.file, 'r')
        files = zip.namelist()
        if len(files) > 5:
            return [['ERROR', 'Maximum 5 files in archive allowed.']]
        if len(set([os.path.splitext(f)[1] for f in files])) > 1:
            return [['ERROR', 'All of the sources have to be written in the same language.']]
        return [(f, zip.read(f).replace('<', '&lt;').replace('>', '&gt;')) for f in files]

    else:
        # Single source file
        text = source.file.read()
        return [[filename, text.replace('<', '&lt;').replace('>', '&gt;')]]


@view_config(route_name='get_example_file', renderer='string')
def get_example_file(request):
    """Return
    """
    # Sanitize file path (probably unnecessary, but it can't hurt)
    tool     = os.path.basename(request.matchdict['tool'])
    filename = os.path.basename(request.matchdict['filename'])

    path = os.path.abspath(os.path.join(request.registry.settings['paws.validation'], 'tools', tool, filename))
    try:
        return file(path).read()
    except:
        return ''
