# -*- coding: utf-8 -*-

import sys, os, re, traceback, shutil
from tempfile            import mkdtemp

from pygments            import highlight
from pygments.lexers     import CLexer, FortranLexer
from pygments.formatters import HtmlFormatter

from pyramid.view        import view_config
from pyramid.renderers   import render

from ..utils             import languages
from ..helpers           import submit
from ..schema            import Params

imports = ''


def _create_file(request, op, code, lang):
    """Create a temp file to receive a source file, and return its full path.
    """
    tmpdir  = request.registry.settings['paws.tempdir']
    fname   = os.path.basename(op + "_code" + languages[lang]['ext']) # sanitize
    workdir = os.path.basename(request.session['workdir']) # sanitize
    path    = os.path.join(tmpdir, workdir, fname)
    file(path, 'w').write(code)
    print "CREATE_FILE", path
    return path

def _get_workdir(request):
    """
    """
    resultdir = request.registry.settings['paws.resultdir']
    workdir   = os.path.basename(request.session['workdir']) # sanitize
    path = os.path.join(resultdir, workdir)
    if not os.path.exists(path): os.mkdir(path)
    return path

def _get_directory(request):
    """Create and return a per-session temporary working directory
    """
    tmpdir  = request.registry.settings['paws.tempdir']
    workdir = mkdtemp(dir=tmpdir)
    request.session['workdir'] = os.path.basename(workdir)
    return request.session['workdir']

def _create_result_file(request, code):
    """
    """
    path = _get_workdir(request)
    file(os.path.join(path, request.session['workdir']), 'w').write(code)

def _create_result_graphs(request, graph):
    """
    """
    path = _get_workdir(request)
    file(os.path.join(path, os.path.basename(graph)), 'w').write(file(graph).read()) ##TODO!!
	
def _delete_dir(filename):
    """Delete the directory containing 'filename'.
    """
    dirname = os.path.dirname(filename)
    if os.path.exists(dirname):
        shutil.rmtree(dirname)

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


def _highlight_code(request, code, lang, demo=False):
    """Apply Pygment formatting
    """
    code = code.replace('\n\n', '\n')
    if not demo:
        _create_result_file(request, code)
    lexer = None
    if lang == 'C':
        lexer = CLexer()
    elif lang in ('Fortran77', 'Fortran95'):
        lexer = FortranLexer()
    if lexer:
        code  = highlight(code, lexer, HtmlFormatter()).replace('<pre>', '<pre>\n')
        lines = [ '<li>%s</li>' % l for l in code.split('\n') if l[:4] != "<div" and l[:5]!="</pre" ]
        return '<pre class="prettyprint linenums"><ol class="highlight linenums">%s</ol></pre>' % ''.join(lines) # absolutely NO blank spaces!


#
# Views
#


@view_config(route_name='get_directory', renderer='string', permission='view')
def get_directory(request):
    """Create and return a per-session temporary working directory
    """
    _get_directory(request)
    # Temporary working directory
    workdir = mkdtemp(dir=request.registry.settings['paws.tempdir'])
    request.session['workdir'] = os.path.basename(workdir)
    return request.session['workdir']


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
    imports = '\n'.join(re.findall(r'^\s*#include.*$', code, re.M)) if lang=='C' else '' # C '#include' lines
    source  = _create_file(request, op, code, lang)
    mod     = _import_base_module(request)
    try:
        functions = mod.perform(source, op, adv, **params)
        _delete_dir(source)
        return _highlight_code(request, imports + '\n' + functions, lang)
    except RuntimeError, msg:
        return _catch_error(msg)
    except:
        return _catch_error('')


