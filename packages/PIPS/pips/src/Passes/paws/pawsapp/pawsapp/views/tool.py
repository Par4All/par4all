# -*- coding: utf-8 -*-

"""
Generic tool controller

"""
import os, re, mimetypes
from ConfigParser import SafeConfigParser
from zipfile      import ZipFile

from pyramid.view import view_config

from .operations  import _get_resdir


mimetypes.init()


def get_tooldir(request, tool):
    """Get tool directory full path.

    :request: Pyramid request
    :tool:    Tool name
    """
    return os.path.join(request.registry.settings['paws.validation'], 'tools', os.path.basename(tool))


def list_examples(request, tool):
    """Get list of examples for the tool.

    :request: Pyramid request
    :tool:    Tool name
    """
    tooldir = get_tooldir(request, tool)
    return [ f for f in sorted(os.listdir(tooldir))
             if os.path.isdir(f)==False and (f.endswith('.c') or f.endswith('.f')) ]


def get_info(request, tool):
    """Get information about the tool.

    :request: Pyramid request
    :tool:    Tool name
    """

    # Callback function 
    def _parse(k, v, sec):
        out = dict(name=k, descr=v[-1])
        if sec.startswith('properties:'):
            if sec == 'properties:bool':
                val = bool(v[0].lower() == 'true')
            elif sec == 'properties:int':
                val = int(v[0])
                out['alt'] = v[1]
            else: # sec == 'properties:str'
                val = v[0:-1]
            out['val'] = val
        return out

    cfg = SafeConfigParser()
    cfg.optionxform = str # case-sensitive keys
    cfg.read(os.path.join(get_tooldir(request, tool), 'info.ini'))

    info = { 'title' : cfg.get('info', 'title'),
             'descr' : cfg.get('info', 'description'),
             }
    for cat in ('properties', 'analyses', 'phases'):        
        info[cat] = { sec.split(':')[1] : [ _parse(k, v.split(';'), sec) for k,v in cfg.items(sec) ]
                      for sec in cfg.sections() if sec.startswith(cat + ':')
                      }
    return info


@view_config(route_name='tool_basic',    renderer='pawsapp:templates/tool.mako', permission='view')
@view_config(route_name='tool_advanced', renderer='pawsapp:templates/tool.mako', permission='view')
def tool(request):
    """Generic tool view (basic and advanced modes).
    """
    tool     = os.path.basename(request.matchdict['tool']) # sanitized
    advanced = bool(request.matched_route.name.endswith('advanced'))

    return dict(tool     = tool,
                info     = get_info(request, tool),
                examples = list_examples(request, tool),
                advanced = advanced,
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
    """Return the content of an example file.
    """
    # Sanitize file path (probably unnecessary, but it can't hurt)
    tool = os.path.basename(request.matchdict['tool'])
    name = os.path.basename(request.matchdict['name'])

    path = os.path.abspath(os.path.join(request.registry.settings['paws.validation'], 'tools', tool, name))
    try:
        return file(path).read()
    except:
        return ''

@view_config(route_name='tool_results',      renderer='string', permission='view')
@view_config(route_name='tool_results_name', renderer='string', permission='view')
def get_result_file(request):
    """Return the content of the transformed code.
    """
    resdir  = _get_resdir(request)
    workdir = os.path.basename(request.session['workdir']) # sanitized
    name    = os.path.basename(request.matchdict.get('name', 'result-%s.txt' % workdir))
    ext     = os.path.splitext(name)[1]
    path    = os.path.join(resdir, name)

    request.response.headers['Content-type'] = mimetypes.types_map.get(ext, 'text/plain;charset=utf-8')

    if ext == '.txt': # Open text file as an attachment
        request.response.headers['Content-disposition'] = str('attachment; filename=%s' % name)

    return file(path).read()

