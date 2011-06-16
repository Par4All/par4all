import logging

from pylons import request, response, session, tmpl_context as c, url
from pylons.controllers.util import abort, redirect

from pawsapp.lib.base import BaseController, render

log = logging.getLogger(__name__)

class PaasController(BaseController):

    def index(self):
        del response.headers['Cache-Control']
	del response.headers['Pragma']

	response.cache_expires(seconds=30)
	return render('/paas.mako')

