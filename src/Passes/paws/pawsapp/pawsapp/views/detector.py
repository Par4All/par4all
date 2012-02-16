# -*- coding: utf-8 -*-

import os, shlex
from subprocess   import Popen, PIPE
from tempfile     import NamedTemporaryFile

from pyramid.view import view_config

from ..utils      import CDecoder, FortranDecoder, languages


def _compile(code, language):
    """Test compilation on a given code snippet as a rough syntax check.
    :code:     source code snippet.
    :language: target language.
    """
    if language not in languages:
        return "127" ##TODO
    langd = languages[language]
    f = NamedTemporaryFile(suffix=langd['ext'])
    objname = os.path.splitext(f.name)[0] + '.o'
    f.write(code)
    cmd = "%s -Wall -o %s -c %s" % (langd['cmd'], objname, f.name) ##TODO
    p = Popen(shlex.split(cmd), stdout=PIPE, stderr=PIPE)
    p.wait()
    f.close()
    if os.path.exists(objname):
        os.remove(objname)
    return '0' if p.returncode == 0 else p.communicate()[1].replace('\n', '<br/>')


def detect_language(code):
    """Detext language of given code.
    """
    analyze = {lexer.name:lexer.analyze(code) for lexer in (CDecoder, FortranDecoder)} ##TODO
    results = sorted(analyze.items(), lambda a,b: cmp(b[1], a[1]))
    lang = 'none'
    if results[0][1] > 10.0 and results[1][1] < 35.0:
        if results[0][0] == 'C':
            lang = 'C'
        elif _compile(code, 'Fortran77') == '0':
            lang = 'Fortran77'
        elif _compile(code, 'Fortran95') == '0':
            lang = 'Fortran95'
        else:
            lang = 'Fortran'
    return lang


@view_config(route_name='detect_language', renderer='string', permission='view')
def detect(request):
    """Heuristics to try and identify source code language.
    """
    return detect_language(request.params['code'])

@view_config(route_name='compile', renderer='string', permission='view')
def compile(request):
    """
    """
    return _compile(request.params['code'], request.params['language'])

