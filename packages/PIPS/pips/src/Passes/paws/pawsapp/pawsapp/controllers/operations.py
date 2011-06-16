import logging, sys, os, traceback

import pawsapp.config.paws as paws

from pyps import workspace, module
from pyrops import pworkspace

from pylons import request, response, session, tmpl_context as c, url
from pylons.controllers.util import abort, redirect

from pawsapp.lib.fileutils import FileUtils
from pawsapp.lib.base import BaseController, render

log = logging.getLogger(__name__)

class OperationsController(BaseController, FileUtils):

    def import_mod(self, operation):
	sys.path.append(paws.modules)
	return __import__('paws_' + operation, None, None, ['__all__'])

    def get_source_file(self, operation, code, language):
	self.imports = self.get_includes(code) if language == "C" else ''
	return self.create_file(operation, code, language)

    def invoke(self, mod, function, workspace):
	try:
		return mod.invoke_function(function, workspace)
	except:
		workspace.close()
		raise

    def perform(self):
	return self.perform_operation(request.params['operation'], request.params['code'], request.params['language'])

    def perform_advanced(self):
	return self.perform_operation(request.params['operation'], request.params['code'], request.params['language'], request.params['type'], request.params['properties'], True)

    def perform_operation(self, operation, code, language, analysis=None, properties=None, advanced=False):
	source_file = self.get_source_file(operation, code, language)
	mod = self.import_mod(operation)
	try:
		ws = pworkspace(str(source_file), deleteOnClose=True)
		if advanced:
			mod.set_properties(ws, str(properties).split(';')[: -1])
			mod.activate(ws, analysis)
		functions = ''
		for fu in ws.fun:
			functions += self.invoke(mod, fu, ws)
		ws.close()
		self.delete_dir(source_file)
		return self.highlight_code(self.imports + '\n' + functions, language)
        except RuntimeError, msg:
                traceback.print_exc(file=sys.stdout)
                traceback_msg = traceback.format_exc()
                return 'EXCEPTION<br/>' + str(msg) + '<br/><br/>TRACEBACK:<br/>' + traceback_msg.replace('\n', '<br/>')
        except:
                traceback.print_exc(file=sys.stdout)
                traceback_msg = traceback.format_exc()
                return 'EXCEPTION<br/><br/>TRACEBACK:<br/>' + traceback_msg.replace('\n', '<br/>')

    def perform_multiple(self):

	function = request.params['function']
	operation = request.params['operation']
	mod = self.import_mod(operation)
	try:
		ws = pworkspace(*session['sources'], deleteOnClose=True)
		result = self.invoke(mod, ws.fun.__getattr__(function), ws)
		ws.close()
		return self.highlight_code(result, 'C')
	except RuntimeError, msg:
		traceback.print_exc(file=sys.stdout)
		traceback_msg = traceback.format_exc()
		return 'EXCEPTION<br/>' + str(msg) + '<br/><br/>TRACEBACK:<br/>' + traceback_msg.replace('\n', '<br/>')
	except:
		traceback.print_exc(file=sys.stdout)
		traceback_msg = traceback.format_exc()
		return 'EXCEPTION<br/><br/>TRACEBACK:<br/>' + traceback_msg.replace('\n', '<br/>')

    def get_directory(self):
        return self.create_directory()

    def get_functions(self):
        sources = []
	print request.params['number']
        for i in range(int(request.params['number'])):
                sources.append(self.create_file('functions' + str(i), request.params['code' + str(i)], request.params['lang' + str(i)]))
	session['sources'] = sources
	session.save()
	print session['sources'][0][0]
        return '\n'.join(['<input value="%s" type="submit"/>' % f for f in self.analyze_functions(sources)])

    def analyze_functions(self, source_files):

        ws = pworkspace(*source_files, deleteOnClose=True)
        functions = []
        for fu in ws.fun:
                functions.append(fu.name)
        ws.close()
	session['methods'] = functions
	session.save()
	print session['methods'][0][0]
        return functions

