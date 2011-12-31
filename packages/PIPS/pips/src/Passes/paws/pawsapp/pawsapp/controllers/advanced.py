import logging, re

import pawsapp.config.paws as paws

from pylons.decorators import jsonify

from pylons import request, response, session, tmpl_context as c, url
from pylons.controllers.util import abort, redirect

from pawsapp.lib.base import BaseController, render

from pyps import workspace, module

log = logging.getLogger(__name__)

class AdvancedController(BaseController):

    @jsonify	
    def load_analyses(self):
	operation = request.params['operation']
	return self.read_analysis_from_file(operation)

    def load_phases(self):
	operation = request.params['operation']
	return self.read_phases_from_file(operation)

    def read_phases_from_file(self, operation):
	phases_types = file(paws.examples + operation + '/phases.lst').read().split('<<')
	text = ''
	for phase in phases_types[1 : ]:
		lines = phase.split('\n')
		text += ''.join([ '<input onChange="enable_performing()" "onMouseOver="popup(\' %s \')" onMouseOut="remove_popup()" name="phases" value="%s" type="checkbox" checked/>%s<br/>' % (t.split(';')[1], t.split(';')[0], t.split(';')[0].upper()) for t in lines[1 : -1] ])	
	return text

    def read_analysis_from_file(self, operation):
	analysis_types = file(paws.examples + operation + '/analyses.lst').read().split('<<')
	analyses = []
	text = ""
	for analysis in analysis_types[1 : ]:
		lines = analysis.split('\n')
		title = lines[0][ : -2]
		analyses.append(title)
		text += '<p><input name="' + title + '" value="' + title + '"onChange="enable_performing()" type="checkbox" checked>' + title + '&nbsp;&nbsp;<select id="analysis_' + title + '">'
		text += ''.join([ '<option onMouseOver="popup(\'%s \')" onMouseOut="remove_popup()" value="%s">%s</option>' % (t.split(';')[-1], t.split(';')[0], t.split(';')[0]) for t in lines[1 : -1] ])	
		text += '</select></p>'
	return [analyses, text]

    def load_properties(self):
	operation = request.params['operation']
	return self.read_properties_from_file(operation)

    def read_properties_from_file(self, operation):
	properties_types = file(paws.examples + operation + '/properties.lst').read().split('<<')
	size = len(properties_types)
	bools = properties_types[1].split('\n')[1:-1] if size > 1 else ""
	ints = properties_types[2].split('\n')[1:-1] if size > 2 else ""
	strs = properties_types[3].split('\n')[1:-1] if size > 3 else ""
	session['int_properties'] = self.get_possible_values(ints)
	session['str_properties'] = self.get_possible_values(strs)
	session.save()
	properties = '<p><b>TRUE/FALSE:</b></p>' if size > 1 else ""
	properties += ''.join([ '<input onChange="enable_performing()" onMouseOver="popup(\' %s \')" onMouseOut="remove_popup()" name="properties_bools" value="%s" type="checkbox" %s/>%s<br/>' % (t.split(';')[2], t.split(';')[0], 'checked' if t.split(';')[1] == 'True' else '',t.split(';')[0]) for t in bools ])
	properties += '<p><b>INTEGER:</b></p>' if size > 2 else ""
	properties += ''.join([ '<input onChange="enable_performing()" onMouseOver="popup(\' %s \')" onMouseOut="remove_popup()" name="properties_ints" value="%s" type="checkbox" checked/>%s&nbsp;&nbsp;&nbsp;<input id="int_%s" type="text" size=5 value="%s"><br/>' % (t.split(';')[-1], t.split(';')[0], t.split(';')[0], t.split(';')[0], t.split(';')[1]) for t in ints ])

	properties += '<p><b>STRING:</b></p>' if size > 3 else ""
	properties += '<div class="ui-widget">'
	properties += ''.join(['<input onChange="enable_performing()" onMouseOver="popup(\' %s \')" onMouseOut="remove_popup()" name="properties_strs" value="%s" type="checkbox" checked/>%s&nbsp;&nbsp;&nbsp;<select id="str_%s">%s</select><br/>' % (t.split(';')[-1], t.split(';')[0], t.split(';')[0], t.split(';')[0], self.get_values(t.split(';')[0])) for t in strs])
	return properties + '</div>'

    def get_values(self, prop):
	return ''.join(['<option value="%s">%s</option>' % (p, p) for p in session['str_properties'][prop]])

    def get_possible_values(self, properties):
	possible_values = {}
	for prop in properties:
		values = prop.split(';')
		possible_values[values[0]] = values[1 : -1]
	return possible_values

    def validate(self):
	properties = request.params['properties'].split(';')
	result = ""
	for prop in properties:
		if prop != "":
			pair = prop.split(" ")
			if pair[0] in session['int_properties']:
				if pair[1] not in session['int_properties'][pair[0]] and '-' not in session['int_properties'][pair[0]]:
					result += "\nProperty " + pair[0] + " has wrong value.\nPossible values: " + str(session['int_properties'][pair[0]])
			elif pair[0] in session['str_properties']:
				if pair[1] not in session['str_properties'][pair[0]] and '-' not in session['str_properties'][pair[0]]:
					result += "\nProperty " + pair[0] + " has wrong value.\nPossible values: " + str(session['str_properties'][pair[0]])
	return result
