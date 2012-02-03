# -*- coding: utf-8 -*-

"""
Generic tool controller

"""
import os, re

from zipfile      import ZipFile
from pyramid.view import view_config


toolName = dict(
    preconditions = u'Preconditions over scalar integer variables',
    openmp        = u'Openmp demo page',
    in_regions    = u'IN regions',
    out_regions   = u'OUT regions',
    regions       = u'Array regions',
    )


def _list_examples(tool, request):
    """Get list of examples for the specified tool.
    """
    path = os.path.join(request.registry.settings['paws.validation'], 'tools', os.path.basename(tool))
    return [ f for f in sorted(os.listdir(path))
             if os.path.isdir(f) == False
             and (f.endswith('.c') or f.endswith('.f'))]


def _parse_lst_file(lst_file, callback):
    """Parse a .lst file
    """
    out = {}
    if os.path.exists(lst_file):
        for line in file(lst_file):
            match = re.match(r'<<(\w+)>>', line)
            if match: # Section header
                section = match.group(1)
                out[section] = []
            else:     # Property line
                p = line.replace('\n', '').split(';')
                out[section].append(callback(p, section))
    return out


def _cb_props(p, section):
    """Callback for property list parsing
    """
    out = { 'name': p[0], 'descr': p[-1] }
    if section == 'bool':
        val = bool(p[1].lower() == 'true')
    elif section == 'int':
        val = int(p[1])
    else: # section == 'str'
        val = p[1:-1]
    out['val'] = val
    return out


def _cb_analyses(p, section):
    """Callback for analyses parsing
    """
    return dict(name=p[0], descr=p[-1])


def _get_props(request):
    """Get properties for tool (advanced mode)
    """
    tool = os.path.basename(request.matchdict['tool']) # sanitized
    return _parse_lst_file( os.path.join(request.registry.settings['paws.validation'], 'tools', tool, 'properties.lst'),
                            callback = _cb_props)

def _get_analyses(request):
    """Get analyses for tool (advanced mode)
    """
    tool = os.path.basename(request.matchdict['tool']) # sanitized
    return _parse_lst_file( os.path.join(request.registry.settings['paws.validation'], 'tools', tool, 'analyses.lst'),
                            callback = _cb_analyses)


def _get_phases(request):
    """Get phases for tool (advanced mode)
    """
    tool = os.path.basename(request.matchdict['tool']) # sanitized
    return _parse_lst_file( os.path.join(request.registry.settings['paws.validation'], 'tools', tool, 'phases.lst'),
                            callback = _cb_analyses)


@view_config(route_name='tool_basic',    renderer='pawsapp:templates/tool.mako', permission='view')
@view_config(route_name='tool_advanced', renderer='pawsapp:templates/tool.mako', permission='view')
def tool(request):
    """Generic tool view (basic and advanced modes).
    """
    tool     = os.path.basename(request.matchdict['tool'])           # (sanitized)
    section  = [ s for s in request.site_sections if s['path']=="tools" ][0]
    entry    = [ e for e in section['entries'] if e['name'] == tool ][0]
    advanced = bool(request.matched_route.name.endswith('advanced'))
    props    = _get_props(request)
    analyses = _get_analyses(request)
    phases   = _get_phases(request)

    return dict(tool     = tool,
                name     = toolName[tool],
                descr    = entry['descr'],
                advanced = advanced,
                props    = props,
                analyses = analyses,
                phases   = phases,
                examples = _list_examples(tool, request),
                )


@view_config(route_name='upload_user_file', renderer='json', permission='view')
def upload_user_file(request):
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
        return [ (f, zip.read(f)) for f in files ]

    else:
        # Single source file
        text = source.file.read()
        return [ (filename, text) ]


@view_config(route_name='load_example_file', renderer='string', permission='view')
def load_example_file(request):
    """Return
    """
    # Sanitize file path (probably unnecessary, but it can't hurt)
    tool = os.path.basename(request.matchdict['tool'])
    name = os.path.basename(request.matchdict['name'])

    path = os.path.abspath(os.path.join(request.registry.settings['paws.validation'], 'tools', tool, name))
    try:
        return file(path).read()
    except:
        return ''
