import logging

from pylons import request, response, session, tmpl_context as c, url
from pylons.controllers.util import abort, redirect

from pawsapp.lib.base import BaseController, render

log = logging.getLogger(__name__)

class ToolsPreconditionsController(BaseController):

    def index(self):
        del response.headers['Cache-Control']
        del response.headers['Pragma']

        response.cache_expires(seconds=20)
        return render('/tools_preconditions.mako')

    def advanced(self):
        del response.headers['Cache-Control']
        del response.headers['Pragma']

        response.cache_expires(seconds=20)
        return render('/tools_preconditions_advanced.mako')
