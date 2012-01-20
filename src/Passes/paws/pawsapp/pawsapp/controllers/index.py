# -*- coding: utf-8 -*-

from pylons import request, response, session, tmpl_context as c, url, config
from pylons.controllers.util import abort, redirect
from pawsapp.lib.base import BaseController, render

import os, json


class IndexController(BaseController):

    # PAWS home page
    def index(self):
        """PAWS home page
        """
        val_path = config['paws.validation']

        # Introductory text
        c.text = file(os.path.join(val_path, 'main', 'paws.txt')).read()

        # Sections
        sections = json.load(file(os.path.join(val_path, 'main', 'functionalities.txt')))
        for s in sections:
            path = os.path.join(val_path, s['path'])            
            s['tools'] = [ { 'name'  : t,
                             'descr' : file(os.path.join(val_path, s['path'], t, '%s.txt' % t)).read(),
                             }
                           for t in os.listdir(path) if not t.startswith('.') 
                           ]
        c.sections = sections

	return render('/index.mako')

