# -*- coding: utf-8 -*-

from pyramid.config  import Configurator
from pyramid.events  import subscriber, BeforeRender
from pyramid.session import UnencryptedCookieSessionFactoryConfig

import helpers


# Paramètres supplémentaires envoyés aux templates
@subscriber(BeforeRender)
def add_global(event):
    """Paramètres supplémentaires envoyés aux templates
    """
    event['h'] = helpers


def main(global_config, **settings):
    """ This function returns a Pyramid WSGI application.
    """
    # Sessions
    session_factory = UnencryptedCookieSessionFactoryConfig(settings['paws.cookie_secret'])

    config = Configurator( settings=settings,
                           session_factory = session_factory,
                           )

    # Static routes
    config.add_static_view('static', 'static', cache_max_age=3600)

    # Dynamic routes
    config.add_route('home', '/')

    config.add_route('tool_basic',       '/tools/{tool}')
    config.add_route('tool_advanced',    '/tools/{tool}/advanced')

    config.add_route('upload_userfile',  '/userfiles/upload')
    config.add_route('get_example_file', '/examples/get_file/{tool}/{filename}')

    config.add_route('detect_language',  '/detector/detect_language')
    config.add_route('compile',          '/detector/compile')

    config.add_route('get_directory',    '/operations/get_directory')
    config.add_route('get_functions',    '/operations/get_functions')
    config.add_route('perform',          '/operations/perform')

    config.scan()
    return config.make_wsgi_app()
