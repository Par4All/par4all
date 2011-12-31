import logging, operator, subprocess, shlex, os

from pylons import request, response, session, tmpl_context as c, url
from pylons.controllers.util import abort, redirect

from pawsapp.lib.base import BaseController, render
from pawsapp.lib.language_detector import *
from pawsapp.lib.fileutils import FileUtils

from pygments import highlight
from pygments.lexers import CLexer
from pygments.formatters import RtfFormatter

from subprocess import PIPE

log = logging.getLogger(__name__)

class DetectorController(BaseController):

    def detect_language(self):
	analyze = {}
	code = request.params['code']
	mod = __import__('pawsapp.lib.language_detector', None, None, ['__all__'])
	for lexer_name in mod.__all__:
		lexer = getattr(mod, lexer_name)()
		analyze[lexer.name] = lexer.analyze(code)
	results = sorted(analyze.iteritems(), key=operator.itemgetter(1), reverse=True)
	print results
	if results[0][1] > 10.0 and results[1][1] < 35.0:
		if results[0][0] == "C":
			return results[0][0]
		if self.compile_language(code, "Fortran77") == '0':
			return "Fortran77"
		if self.compile_language(code, "Fortran95") == '0':
			return "Fortran95"
		else:
			return "Fortran"
	else:
		return "none"

    def compile_language(self, code, language):
	filename = 'user_code'
	f = open(filename + FileUtils.extensions[language], 'w')
	f.write(code)
	f.close()
	languages = {
		"C": 'gcc',
		"Fortran77": 'f77',
		"Fortran95": 'gfortran',
		"Fortran": 'f77'
	}
	cmd = languages.get(language, None) + ' -Wall -o ' + filename + '.o -c ' + f.name
	p = subprocess.Popen(shlex.split(cmd), stdout = PIPE, stderr = PIPE)
	p.wait()
	os.remove(f.name)
	if os.path.exists(filename + ".o"): os.remove(filename + ".o")
	if p.returncode == 0:
		return '0'
	return p.communicate()[1].replace('\n', '<br/>')
	
    def compile(self):
	return self.compile_language(request.params['code'], request.params['language'])
	
