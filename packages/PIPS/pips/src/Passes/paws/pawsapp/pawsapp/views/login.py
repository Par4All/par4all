# -*- coding: utf-8 -*-

import os

from pyramid.httpexceptions import HTTPFound
from pyramid.security       import remember, forget
from pyramid.renderers      import render_to_response
from pyramid.view           import view_config


from ..security import USERS


@view_config(route_name='login', renderer='pawsapp:templates/login.mako')
def login(request):
    """
    """
    # Introductory text
    valid_path = request.registry.settings['paws.validation']
    text = file(os.path.join(valid_path, 'main', 'paws.txt')).read()

    login_url = request.route_url('login')
    referrer  = request.url
    if referrer == login_url:
        referrer = '/' # never use the login form itself as came_from

    came_from = request.params.get('came_from', referrer)
    message = ''
    login = ''
    password = ''

    if 'form.submitted' in request.params:
        login    = request.params['login']
        password = request.params['password']

        if USERS.get(login) == password:
            headers = remember(request, login)
            return HTTPFound( location = came_from,
                              headers = headers)
        else:
            message = 'Failed login'

    return dict( text      = text,    # intro text
                 message   = message,
                 url       = request.application_url + '/login',
                 came_from = came_from,
                 login     = login,
                 password  = password,
                 )


@view_config(route_name='logout')
def logout(request):
    """
    """
    headers = forget(request)
    request.session.invalidate()
    return HTTPFound(location = request.route_url('home'), headers = headers)
    
