import os, subprocess, shlex
from subprocess import PIPE
from pylons import request, response, session, tmpl_context as c, url
from pylons.controllers.util import abort, redirect
from pawsapp.lib.base import BaseController, render

from pawsapp.lib.fileutils import FileUtils


class ExamplesController(BaseController, FileUtils):

    def demo(self):
	filename = request.params['name']
	return self.get_file_content(request.params['operation'], filename[ : filename.index('.')] + '.tpips')
    
    def perform(self):
	name = request.params['name']
	return self.run_tpips(paws.examples + request.params['operation'] + '/' + name[: name.rindex('.')], name[name.rindex('.') + 1])

    def run_tpips(self, filename, extension):
	p = subprocess.Popen(['tpips', filename + '.tpips', '1'], stdout = PIPE, stderr = PIPE)
	p.wait()
	result = p.communicate()[0]
	if result == self.get_example_result(filename):
		language = "C" if extension == "c" else "Fortran95"
		return self.highlight_code(result, language)
	else:
		return "Error while performing operation - wrong output!"

    def get_example_result(self, filename):
	return file(filename + '.result/test').read()
	

