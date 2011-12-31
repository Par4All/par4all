import logging, sys, os, traceback

import pawsapp.config.paws as paws

from pylons import request, response, session, tmpl_context as c, url
from pylons.controllers.util import abort, redirect

from pawsapp.lib.fileutils import FileUtils
from pawsapp.lib.base import BaseController, render

log = logging.getLogger(__name__)

class OperationsController(BaseController, FileUtils):

    def import_base_module(self):
	sys.path.append(paws.modules)
	return __import__('paws_base', None, None, ['__all__'])

    def get_source_file(self, operation, code, language):
	self.imports = self.get_includes(code) if language == "C" else ''
	return self.create_file(operation, code, language)

    def perform(self):
	return self.invoke_module(request.params['operation'], request.params['code'], request.params['language'])

    def perform_advanced(self):
	return self.invoke_module(request.params['operation'], request.params['code'], request.params['language'], request.params['analyses'], request.params['properties'], request.params['phases'], True)

    def invoke_module(self, operation, code, language, analysis=None, properties=None, phases=None, advanced=False):
	source_file = self.get_source_file(operation, code, language)
	mod = self.import_base_module()
	try:
		functions = mod.perform(os.getcwd() + '/' + str(source_file), operation, advanced, properties, analysis, phases)
		self.delete_dir(source_file)
		return self.highlight_code(self.imports + '\n' + functions, language)
        except RuntimeError, msg:
		return self.catch_error(msg)
        except:
		return self.catch_error('')

    def invoke_module_multiple(self, sources, operation, function, language, analysis=None, properties=None, phases=None, advanced=False):
	mod = self.import_base_module()
	try:
		result = mod.perform_multiple(sources, operation, function, advanced, properties, analysis, phases)
		return self.highlight_code(result, language)
        except RuntimeError, msg:
		return self.catch_error(msg)
        except:
		return self.catch_error('')

    def perform_multiple(self):
	return self.invoke_module_multiple(session['sources'], request.params['operation'], request.params['functions'], request.params['language'])

    def perform_multiple_advanced(self):
	return self.invoke_module_multiple(session['sources'], request.params['operation'], request.params['functions'], request.params['language'], request.params['analyses'], request.params['properties'], request.params['phases'], True)

    def catch_error(self, msg):
	traceback.print_exc(file=sys.stdout)
        traceback_msg = traceback.format_exc()
        return 'EXCEPTION<br/>' + str(msg) + '<br/><br/>TRACEBACK:<br/>' + traceback_msg.replace('\n', '<br/>')

    def get_directory(self):
        return self.create_directory()

    def get_functions(self):
        sources = []
        for i in range(int(request.params['number'])):
                sources.append(self.create_file('functions' + str(i), request.params['code' + str(i)], request.params['lang' + str(i)]))
	session['sources'] = sources
	session.save()
	print session['sources'][0][0]
        return '\n'.join(['<input value="%s" type="submit"/>' % f for f in self.analyze_functions(sources)])

    def analyze_functions(self, source_files):

	mod = self.import_base_module()
	session['methods'] = mod.get_functions(source_files)
	session.save()
        return session['methods']

