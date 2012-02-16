# -*- coding: utf-8 -*-

import sys, os, re, traceback, shutil, mimetypes
from tempfile            import mkdtemp

from pyramid.view        import view_config
from pyramid.renderers   import render

from ..utils             import languages, htmlHighlight
from ..helpers           import submit
from ..schema            import Params

mimetypes.init()

_resultDirName = '__res__'
imports        = ''


def create_workdir(request):
    """Create a per-session temporary working directory, update session and return its base name.
    """
    tempdir = request.registry.settings['paws.tempdir']
    workdir = mkdtemp(dir = tempdir)
    os.mkdir(os.path.join(workdir, _resultDirName))
    dirname = os.path.basename(workdir)
    request.session['workdir'] = dirname
    return dirname
    

def get_workdir(request, reuse=True):
    """Create a per-session temporary working directory and return its base name.
    Clear previous workdir files, if any, unless reuse is OK.

    :request:  Pyramid request
    :reuse:    In case workdir already exists, can we reuse it?
    """
    tempdir = request.registry.settings['paws.tempdir']

    if 'workdir' in request.session:
        dirname = os.path.basename(request.session['workdir']) # sanitized
        path    = os.path.join(tempdir, dirname)
        if os.path.exists(path):
            if not reuse:
                shutil.rmtree(path)

        if not os.path.exists(path):
            # workdir should be here, but missing for some reason, or if was deleted just above.
            os.mkdir(path)
        if not os.path.exists(os.path.join(path, _resultDirName)):
            # same as above
            os.mkdir(os.path.join(path, _resultDirName))

    else:
        # create brand new workdir
        dirname = create_workdir(request)

    return dirname


def get_resultdir(request):
    """Return fullpath to temporary results directory.
    """
    tempdir = request.registry.settings['paws.tempdir']
    workdir = os.path.basename(request.session['workdir']) # sanitized
    return os.path.join(tempdir, workdir, _resultDirName)

def _create_file(request, op, code, lang):
    """Create a temp file to receive a source file, and return its full path.
    """
    tempdir = request.registry.settings['paws.tempdir']
    workdir = os.path.basename(request.session['workdir']) # sanitized
    fname   = os.path.basename(op + "_code" + languages[lang]['ext']) # sanitized
    path    = os.path.join(tempdir, workdir, fname)
    file(path, 'w').write(code)
    return path

def _import_base_module(request):
    """
    """
    sys.path.append(os.path.join(request.registry.settings['paws.validation'], 'pyps_modules')) ##TODO
    return __import__('paws_base', None, None, ['__all__'])

def _analyze_functions(request, source_files):
    """
    """
    mod = _import_base_module(request)
    request.session['methods'] = mod.get_functions(source_files)
    return request.session['methods']

def _catch_error(msg):
    traceback.print_exc(file=sys.stdout)
    traceback_msg = traceback.format_exc()
    return 'EXCEPTION<br/>' + str(msg) + '<br/><br/>TRACEBACK:<br/>' + traceback_msg.replace('\n', '<br/>')

def _invoke_module_multiple( request, sources, operation, function, lang,
                             analysis=None, properties=None, phases=None,
                             advanced=False):
    """
    """
    mod = _import_base_module(request)
    try:
        result = _perform_multiple(sources, operation, function, advanced, properties, analysis, phases)
        return _highlight_code(result, lang)
    except RuntimeError, msg:
        return _catch_error(msg)
    except:
        return _catch_error('')


def _perform_multiple(request):
    form = request.params
    return _invoke_module_multiple( request.session['sources'], form['operation'],
                                    form['functions'], form['language'])

def perform_multiple_advanced(request):
    form = request.params
    return _invoke_module_multiple( request.session['sources'], form['operation'],
                                    form['functions'], form['language'], form['analyses'],
                                    form['properties'], form['phases'], True)


def _write_result_code(request, code):
    """Write temporary result code file to disk.
    """
    resdir  = get_resultdir(request)
    workdir = os.path.basename(request.session['workdir']) # Sanitized
    path    = os.path.join(resdir, 'result-%s.txt' % workdir)
    file(path, 'w').write(code)


#
# Views
#


@view_config(route_name='get_directory', renderer='string', permission='view')
def get_directory(request):
    """Create and return a per-session temporary working directory
    """
    return get_workdir(request, reuse=False)


@view_config(route_name='get_functions', renderer='string', permission='view')
def get_functions(request):
    """
    """
    form    = request.params
    sources = [ _create_file(request, 'functions%d' % i, form['code%d' % i],  form['lang%d' % i])
                for i in range(int(form['number']))]
    request.session['sources'] = sources
    print request.session['sources'][0][0]
    return '\n'.join([ submit(f, f, class_='btn small') for f in _analyze_functions(request, sources) ])


@view_config(route_name='perform', renderer='string', permission='view')
def perform(request):
    """Perform operation (basic/advanced mode)
    """
    form = request.params
    op   = form.get('op')
    code = form.get('code')
    lang = form.get('lang')
    adv  = bool(form.get('adv') == 'true')

    if adv:
        # Advanced mode: deserialize form values
        schema = Params()
        data   = dict(p.split('=') for p in form.getall('params[]'))
        params = schema.deserialize(schema.unflatten(data))
    else:
        # Basic mode
        params = {}

    # Perform operation
    global imports ##TODO
    imports = '\n'.join(re.findall(r'^\s*#include.*$', code, re.M)) if lang=='C' else '' # C include directives
    source  = _create_file(request, op, code, lang)
    mod     = _import_base_module(request)
    try:
        funcs = mod.perform(source, op, adv, **params)
        code  = (imports + '\n' + funcs).replace('\n\n', '\n')
        _write_result_code(request, code)
        return htmlHighlight(code, lang)
    except RuntimeError, msg:
        return _catch_error(msg)
    except:
        return _catch_error('')


@view_config(route_name='results',      renderer='string', permission='view')
@view_config(route_name='results_name', renderer='string', permission='view')
def get_result_file(request):
    """Return the content of the transformed code.
    """
    resdir  = get_resultdir(request)
    workdir = os.path.basename(request.session['workdir']) # sanitized
    name    = os.path.basename(request.matchdict.get('name', 'result-%s.txt' % workdir))
    ext     = os.path.splitext(name)[1]
    path    = os.path.join(resdir, name)

    request.response.headers['Content-type'] = mimetypes.types_map.get(ext, 'text/plain;charset=utf-8')

    if ext == '.txt': # Open text file as an attachment
        request.response.headers['Content-disposition'] = str('attachment; filename=%s' % name)

    return file(path).read()

