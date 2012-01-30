import logging, os, subprocess, re, shutil

import pawsapp.config.paws as paws

from pylons import request, response, session, tmpl_context as c, url
from pylons.controllers.util import abort, redirect
from pylons.decorators import jsonify

from pawsapp.lib.base import BaseController, render
from pawsapp.lib.fileutils import FileUtils 
from pawsapp.lib.images import Images

from subprocess import PIPE

log = logging.getLogger(__name__)

class DemoController(BaseController, FileUtils):

	
    def index(self):
	del response.headers['Cache-Control']
	del response.headers['Pragma']
	
	response.cache_expires(seconds=10)
        return render('/demo.mako')
    
    def change_source(self):
	self.create_temporary_source_file(paws.demo + self.get_directory_name(request.params['name']) + '/' + request.params['name'], request.params['code'])
	self.initialize(paws.demo + self.get_directory_name(request.params['name']) + '/' + self.get_tpips_from_name(request.params['name']))
	session['source'] = request.params['code']
	session.save()
	self.restore_input(paws.demo + self.get_directory_name(request.params['name']) + '/' + request.params['name'])
        return str(len(session['operations']) - 1)
     
    def create_temporary_source_file(self, name, code):
        os.rename(name, name + '_')
	file(name, 'w').write(code)
    
    def restore_input(self, name):
        os.rename(name + '_', name)
   
    def get_step_output(self):
	if int(request.params['step']) in session['graphs'].keys():
		return session['results'][int(request.params['step'])]
	language = "C" if request.params['name'].endswith('.c') else "Fortran77"
	return self.highlight_code(''.join(['%s\n' % x for x in (session['results'][int(request.params['step'])]).split('\n')]), language, True) if request.params['step'] != '0' else session['source']
    
    def get_step_script(self):
	return self.highlight_code(''.join(['%s\n' % x for x in session['operations'][int(request.params['step'])] if x.find(paws.marker) == -1 and not x.startswith('echo')]), "C", True) if request.params['step'] != '0' else session['tpips']
 
    def delete_files(self, tpips):
	if os.path.exists(tpips[tpips.rindex('/') + 1 : tpips.rindex('.')] + '.database'):
		shutil.rmtree(tpips[tpips.rindex('/') + 1 : tpips.rindex('.')] + '.database')
	if os.path.exists(session['directory'] + '.png'):
		os.remove(session['directory'] + '.png')
	
    def create_png(self, function, demo_name):
	p = subprocess.Popen(['dot', '-Tpng', '-o', session['directory'] + '.png', demo_name + '.database/' + function + '/' + function + '.dot'], stdout = PIPE, stderr = PIPE)
	p.wait()
	self.create_result_graphs(session['directory'] + '.png')
	return Images().create_zoom_image(session['directory'] + '/' + session['directory'] + '.png')
	
