# -*- coding: utf-8 -*-

import os, json

from pyramid.view import view_config

# PAWS home page
@view_config(route_name='home', renderer='pawsapp:templates/index.mako')
def index(request):
    """PAWS home page
    """
    valid_path = request.registry.settings['paws.validation']

    # Introductory text
    text = file(os.path.join(valid_path, 'main', 'paws.txt')).read()

    # Sections
    sections = json.load(file(os.path.join(valid_path, 'main', 'functionalities.txt')))
    for s in sections:
        path = os.path.join(valid_path, s['path'])            
        s['tools'] = [ { 'name'  : t,
                         'descr' : file(os.path.join(valid_path, s['path'], t, '%s.txt' % t)).read(),
                         }
                       for t in os.listdir(path) if not t.startswith('.') 
                       ]

    return dict(text=text, sections = sections)

