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

    dependence_graph = 'apply PRINT_DOT_DEPENDENCE_GRAPH'
	
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
	p = subprocess.Popen(['mv', name, name + '_'], stdout=PIPE, stderr=PIPE)
	p.wait()
	f = open(name, 'w')
	f.write(code)
	f.close()
    
    def restore_input(self, name):
	p = subprocess.Popen(['mv', name + '_', name], stdout=PIPE, stderr=PIPE)
	p.wait()
   
    def get_steps_number(self):
	session.clear()
	session.save()
	self.initialize(paws.demo + self.get_directory_name(request.params['name']) + '/' + self.get_tpips_from_name(request.params['name']))
        return str(len(session['operations']) - 1)
    
    @jsonify
    def load_demo(self):
       	session['tpips'] = self.highlight_code(re.sub("(echo.*\n)", "", self.get_file_content(self.get_tpips_from_name(request.params['name']))), "C", True)
        session['source'] = self.get_file_content(request.params['name'])
	session.save()
	return [session['tpips'], session['source']]
     
    def get_directory_name(self, name):
	return name[ : name.index('.')]
	   
    def get_tpips_from_name(self, name):
	return self.get_directory_name(name) + '.tpips'
	  
    def get_file_content(self, name):
        return file(paws.demo + self.get_directory_name(name) + '/' + name).read()
   
    def get_step_output(self):
	if int(request.params['step']) in session['graphs'].keys():
		return session['results'][int(request.params['step'])]
	language = "C" if request.params['name'].endswith('.c') else "Fortran77"
	return self.highlight_code(''.join(['%s\n' % x for x in (session['results'][int(request.params['step'])]).split('\n')]), language, True) if request.params['step'] != '0' else session['source']
    
    def get_step_script(self):
	return self.highlight_code(''.join(['%s\n' % x for x in session['operations'][int(request.params['step'])] if x.find(paws.marker) == -1 and not x.startswith('echo')]), "C", True) if request.params['step'] != '0' else session['tpips']
 
    def initialize(self, tpips):
	self.operations = {}
	self.results = {}
	self.graphs = {}
	self.create_directory()
	self.parse_tpips(tpips)
	self.create_results(tpips[tpips.rindex('/') + 1 : tpips.rindex('.')])
	self.delete_files(tpips)
	session['operations'] = self.operations
	session['results'] = self.results
	session['graphs'] = self.graphs
	session.save()

    def delete_files(self, tpips):
	if os.path.exists(tpips[tpips.rindex('/') + 1 : tpips.rindex('.')] + '.database'):
		shutil.rmtree(tpips[tpips.rindex('/') + 1 : tpips.rindex('.')] + '.database')
	if os.path.exists(session['directory'] + '.png'):
		os.remove(session['directory'] + '.png')
	
    def parse_tpips(self, tpips):
        text = file(tpips).read().split('\n')
	index = 1
	for line in text:
		if index not in self.operations:
			self.operations[index] = []
		self.operations[index].append(line)
		if line.startswith('display') or line.startswith(self.dependence_graph):
			self.operations[index].append('echo ' + paws.marker + '\n')
			index += 1
			if line.startswith(self.dependence_graph):
				self.graphs[index - 1] = line[line.find('[') + 1 : line.find(']')]

    def create_results(self, demo_name):
        script = self.create_script(demo_name)
	p = subprocess.Popen(['tpips', script, '1'], stdout = PIPE, stderr = PIPE)
	p.wait()
	result = p.communicate()[0].split(paws.marker)
	for index in range(1, len(result) + 1):
		self.results[index] = result[index - 1]
	for index in self.graphs.keys():
		self.results[index] = self.create_png(self.graphs[index], demo_name) 
	os.remove(script)

    def create_png(self, function, demo_name):
	p = subprocess.Popen(['dot', '-Tpng', '-o', session['directory'] + '.png', demo_name + '.database/' + function + '/' + function + '.dot'], stdout = PIPE, stderr = PIPE)
	p.wait()
	self.create_result_graphs(session['directory'] + '.png')
	return Images().create_zoom_image(session['directory'] + '/' + session['directory'] + '.png')
	
    def create_script(self, demo_name):
        f = open(paws.demo + demo_name + '/temporary.tpips', 'w')
	for step in self.operations.keys():
		for line in self.operations[step]:
			f.write(line + '\n')
	f.close()
	return f.name
