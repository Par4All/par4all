import logging, re

import pawsapp.config.paws as paws

from pylons.decorators import jsonify

from pylons import request, response, session, tmpl_context as c, url
from pylons.controllers.util import abort, redirect

from pawsapp.lib.base import BaseController, render

from pyps import workspace, module

log = logging.getLogger(__name__)

class AdvancedController(BaseController):

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
