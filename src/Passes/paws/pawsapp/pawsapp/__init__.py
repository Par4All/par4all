# -*- coding: utf-8 -*-

from pyramid.config         import Configurator
from pyramid.authentication import AuthTktAuthenticationPolicy
from pyramid.authorization  import ACLAuthorizationPolicy
from pyramid.security       import authenticated_userid
from pyramid.events         import subscriber, BeforeRender
from pyramid.session        import UnencryptedCookieSessionFactoryConfig

from   security             import groupfinder
from   utils                import getSiteSections
import helpers



# Paramètres supplémentaires envoyés aux templates
@subscriber(BeforeRender)
def add_global(event):
    """Paramètres supplémentaires envoyés aux templates
    """
    request = event['request']
    userid  = authenticated_userid(request)

    event['h']      = helpers
    event['userid'] = userid


def main(global_config, **settings):
    """ This function returns a Pyramid WSGI application.
    """
    # Security
    authn_policy = AuthTktAuthenticationPolicy(settings['paws.authn_secret'], callback=groupfinder)
    authz_policy = ACLAuthorizationPolicy()

    # Sessions
    session_factory = UnencryptedCookieSessionFactoryConfig(settings['paws.cookie_secret'])

    config = Configurator( settings              = settings,
                           root_factory          = 'pawsapp.security.RootFactory',
                           authentication_policy = authn_policy,
                           authorization_policy  = authz_policy,
                           session_factory       = session_factory,
                           )

    # Request properties
    config.set_request_property(getSiteSections, 'site_sections', reify=True)

    # Static routes
    config.add_static_view('static', 'static', cache_max_age=3600)

    # Dynamic routes
    config.add_route('home',                    '/')
    config.add_route('login',                   '/login')
    config.add_route('logout',                  '/logout')

    config.add_route('tool_basic',              '/tools/{tool}')
    config.add_route('tool_advanced',           '/tools/{tool}/advanced')

    config.add_route('tutorial',                '/tutorials/{tutorial}')
    config.add_route('tutorial_init',           '/tutorials/{tutorial}/init')

    config.add_route('results',                 '/results')
    config.add_route('results_name',            '/results/{name}')

    config.add_route('upload_user_file',        '/files/upload')
    config.add_route('load_example_file',       '/files/{tool}/{name}')
    config.add_route('source_panel',            '/util/source_panel/{index}')

    config.add_route('detect_language',         '/detector/detect_language')
    config.add_route('compile',                 '/detector/compile')

    config.add_route('get_directory',           '/operations/get_directory')
    config.add_route('get_functions',           '/operations/get_functions')

    config.add_route('perform',                 '/operations/perform')
    config.add_route('perform_multiple',        '/operations/perform_multiple')

    config.add_route('dependence_graph',        '/graph/dependence_graph')
    config.add_route('dependence_graph_multi',  '/graph/dependence_graph_multi') ##TODO

    config.add_route('routes.js',               '/routes.js')

    ## Special views
    config.add_view('pawsapp.views.login.login',
                    context='pyramid.httpexceptions.HTTPForbidden',
                    renderer='pawsapp:templates/login.mako')

    config.scan()

    return config.make_wsgi_app()
