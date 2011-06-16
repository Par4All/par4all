import logging, os

import pawsapp.config.paws as paws

from pylons.decorators import jsonify

from pylons import request, response, session, tmpl_context as c, url
from pylons.controllers.util import abort, redirect

from pawsapp.lib.base import BaseController, render

log = logging.getLogger(__name__)

class DescriptionsController(BaseController):

    def paws_text(self):
        return file(paws.descriptions + 'paws.txt').read()

    def parse_functions(self, functions):
        dictionary = {}
	two_level = []
	for f in functions:
		func = f.split(':')
		dictionary[func[0]] = func[1]
		if func[2] == '2':
			two_level.append(func[0])
	session['functions'] = dictionary
	session['level'] = two_level
	session.save()
 
    @jsonify
    def sections(self):
	functions = file(paws.descriptions + 'functionalities.txt').read().rstrip().split('\n')
	self.parse_functions(functions)
	print 'dic', session['functions']
	return [f.split(':')[1] for f in functions]

    def accordion(self):
        source = '<table>'
	for section in sorted(session['functions'].keys()):
		directory = session['functions'][section]
		source += '<tr><td width="200px" valign="top"><h3>' + section + ':</h3></td><td width="500px" valign="top"><div id="' + directory + '" width="100%">'
		path = paws.validation + directory
		for tool in os.listdir(path):
			if not tool.startswith('.'):
				source += '<h3><a href="#">' + tool.upper() + '</a></h3><div height="500px"><h5>' + file(path + '/' + tool + '/' + tool + '.txt').read() + '<br/><br/><br/><b>{0}</b></h5></div>'.format(self.create_links(section, directory, tool))
		source += '</div><tr height="30px"><td></td><td></td></tr>'
	source += '</table>'
	return source

    def create_links(self, section, directory, tool):
	if section in session['level']:
        	return '<a href="/{0}_{1}/index">basic</a>&nbsp;&nbsp;<a href="/{0}_{1}/advanced">advanced</a>'.format(directory, tool)
	else:
		return '<a href="/{0}_{1}/index">{1}</a>'.format(directory, tool)
	
