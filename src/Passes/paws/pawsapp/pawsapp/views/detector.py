# -*- coding: utf-8 -*-

import os, shlex, operator
from subprocess          import Popen
from tempfile            import NamedTemporaryFile

from pygments            import highlight
from pygments.lexers     import CLexer
from pygments.formatters import RtfFormatter

from pyramid.view        import view_config

from ..lib.language_detector import *
from ..lib.fileutils         import FileUtils

from subprocess import PIPE

languages = {
    "C"         : 'gcc',
    "Fortran77" : 'f77',
    "Fortran95" : 'gfortran',
    "Fortran"   : 'f77'
    }

def _compile_language(code, language):
    """Test compilation on a given code snippet as a rough syntax check.
    :code:     source code snippet.
    :language: target language.
    """
    f = NamedTemporaryFile(suffix=FileUtils.extensions[language])
    filename = os.path.splitext(f.name)[0]
    f.write(code)
    cmd = languages.get(language, None) + ' -Wall -o ' + filename + '.o -c ' + f.name
    p   = Popen(shlex.split(cmd), stdout = PIPE, stderr = PIPE)
    p.wait()
    f.close()
    if os.path.exists(filename + ".o"):
        os.remove(filename + ".o")
    return '0' if p.returncode == 0 else p.communicate()[1].replace('\n', '<br/>')


@view_config(route_name='detect_language', renderer='string')
def detect_language(request):
    """
    """
    analyze = {}
    code = request.params['code']

    mod = __import__('pawsapp.lib.language_detector', None, None, ['__all__']) ##TODO
    for lexer_name in mod.__all__:
        lexer = getattr(mod, lexer_name)()
        analyze[lexer.name] = lexer.analyze(code)

    results = sorted(analyze.iteritems(), key=operator.itemgetter(1), reverse=True)
    print results

    if results[0][1] > 10.0 and results[1][1] < 35.0:
        if results[0][0] == "C":
            return "C"
        if _compile_language(code, "Fortran77") == '0':
            return "Fortran77"
        if _compile_language(code, "Fortran95") == '0':
            return "Fortran95"
        else:
            return "Fortran"
    else:
        return "none"


def compile(request):
    return _compile_language(request.params['code'], request.params['language'])

