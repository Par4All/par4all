import logging, zipfile, json

from json import dumps

from pylons import request, response, session, tmpl_context as c, url
from pylons.controllers.util import abort, redirect

from pawsapp.lib.base import BaseController, render

log = logging.getLogger(__name__)

class UserfilesController(BaseController):

    def upload(self):
        source = request.POST['file']
	filename = request.POST['file'].filename
	if str(filename).endswith('.zip'):
		zipp = zipfile.ZipFile(source.file, 'r')
		if len(zipp.namelist()) > 5:
			return dumps([['ERROR', 'Maximum 5 files in archive allowed.']])
		extension = zipp.namelist()[0][ zipp.namelist()[0].rindex('.') : ]
		for f in zipp.namelist():
			if f[ f.rindex('.') : ] != extension:
				return dumps([['ERROR', 'All of the sources have to be written in the same language.']])
		return dumps([(f, zipp.read(f).replace('<', '&lt;').replace('>', '&gt;')) for f in zipp.namelist()])
	else:
		text = request.POST['file'].file.read()
		return dumps([[filename, text.replace('<', '&lt;').replace('>', '&gt;')]])

