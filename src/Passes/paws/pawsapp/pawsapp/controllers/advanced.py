import logging, re

from pylons import request, response, session, tmpl_context as c, url
from pylons.controllers.util import abort, redirect

from pawsapp.lib.base import BaseController, render

from pyps import workspace, module

log = logging.getLogger(__name__)

class AdvancedController(BaseController):

    def index(self):
        # Return a rendered template
        #return render('/advanced.mako')
        # or, return a string
        return 'Hello World'

    def load_buttons(self):
	op = request.params['operation']
	return ''.join([ '<input value="%s" type="submit"/><br/>' % t for t in self.read_modules_from_tex('> MODULE.' + op) ])

    def read_modules_from_tex(self, search):
	p = re.compile(search)
	types = []
	f = open('pawsapp/public/pipsmake-rc.tex', 'r')
	for line in f:
		if re.search(search, line):
			types.append(line.split()[0].replace('_', ' '))
	f.close()
	return types

    def load_properties(self):
	op = request.params['operation']
	pr= self.read_properties_from_file(op)
	return pr

    def read_properties_from_file(self, operation):
	properties_types = file(operation + '_properties_list.lst').read().split('<')
	bools = properties_types[1].split('\n')[1:-1]
	ints = properties_types[2].split('\n')[1:-1]
	strs = properties_types[3].split('\n')[1:-1]
	properties = '<p><b>TRUE/FALSE:</b></p>'
	properties += ''.join([ '<input name="properties_bools" value="%s" type="checkbox" %s/>%s<br/>' % (t.split()[0], 'checked' if t.split()[1] == 'True' else '',t.split()[0]) for t in bools ])
	properties += '<p><b>INTEGER:</b></p>'
	properties += ''.join([ '<input name="properties_ints" value="%s" type="checkbox"/>%s&nbsp;&nbsp;&nbsp;<input id="int_%s" type="text" size=5 value="%s"><br/>' % (t.split()[0], t.split()[0], t.split()[0], t.split()[1]) for t in ints ])
	properties += '<p><b>STRING:</b></p>'
	return properties + ''.join([ '<input name="properties_strs" value="%s" type="checkbox"/>%s&nbsp;&nbsp;&nbsp;<input id="str_%s" type="text" size=10 value="%s"><br/>' % (t.split()[0], t.split()[0], t.split()[0], t.split()[1]) for t in strs ])

