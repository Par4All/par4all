# -*- coding: utf-8 -*-

from pyramid.config import Configurator
from pyramid.events import subscriber, BeforeRender

from pawsapp.lib    import helpers


# Paramètres supplémentaires envoyés aux templates
@subscriber(BeforeRender)
def add_global(event):
    """Paramètres supplémentaires envoyés aux templates
    """
    event['h'] = helpers


def main(global_config, **settings):
    """ This function returns a Pyramid WSGI application.
    """
    config = Configurator(settings=settings)

    # Static routes
    config.add_static_view('static', 'static', cache_max_age=3600)

    # Dynamic routes
    config.add_route('home', '/')

    config.scan()
    return config.make_wsgi_app()
