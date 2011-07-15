import logging
import shutil, pyrops, subprocess

from pyps import workspace, module
from pyrops import pworkspace

from subprocess import PIPE

from pylons import request, response, session, tmpl_context as c, url
from pylons.controllers.util import abort, redirect

from pawsapp.lib.fileutils import FileUtils
from pawsapp.lib.base import BaseController, render
from pawsapp.lib.images import Images

log = logging.getLogger(__name__)

class GraphController(BaseController, FileUtils, Images):

    def dependence_graph(self):
	source_file = self.create_file('', request.params['code'], request.params['language'])
        ws = pworkspace(str(source_file), name=session['directory'], deleteOnClose=True)
	functions = []
	images = ''
	for fu in ws.fun:
		try:
			functions.append(fu.name)
			fu.print_dot_dependence_graph()
		except:
			ws.close()
			raise
	for fu in functions:
		filename = 'files/' + session['directory'] + '/' + fu + '.png'
		p = subprocess.Popen(['dot', '-Tpng', '-o', filename, session['directory'] + '.database/' + fu + '/' + fu + '.dot'], stdout = PIPE, stderr = PIPE)
		p.wait()
		self.create_result_graphs(filename)
		images += '<div style="clear: both;width:100%"><b>' + fu + ':</b><br/>' + self.create_zoom_image(session['directory'] + '/' + fu + '.png') + '</div>'
	self.delete_dir(source_file)
	ws.close()
	return images

    def dependence_graph_multi(self):
	sources = session['sources']
	print sources[0][0]
	ws = pworkspace(*sources, name=session['directory'], deleteOnClose=True)
	functions = []
	images = ''
	for fu in ws.fun:
		try:
			functions.append(fu.name)
			fu.print_dot_dependence_graph()
		except:
			ws.close()
			raise
	for fu in functions:
		filename = 'files/' + session['directory'] + '/' + fu + '.png'
		p = subprocess.Popen(['dot', '-Tpng', '-o', filename, session['directory'] + '.database/' + fu + '/' + fu + '.dot'], stdout = PIPE, stderr = PIPE)
		p.wait()
		self.create_result_graphs(filename)
		images += '<div style="clear: both;width:100%"><b>' + fu + ':</b><br/>' + self.create_zoom_image(session['directory'] + '/' + fu + '.png') + '</div>'
	ws.close()
	return images
