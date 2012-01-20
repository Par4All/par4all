try:
    from setuptools import setup, find_packages
except ImportError:
    from ez_setup import use_setuptools
    use_setuptools()
    from setuptools import setup, find_packages

setup(
    name='pawsapp',
    version='0.1',
    description='Web Interface for PIPS',
    author='Maria SZYMCZAK',
    author_email='',
    url='http://paws.pips4u.org',
    install_requires=[
        "Pylons>=1.0",
        "pyro",
        "pil",
    ],
    setup_requires=["PasteScript>=1.6.3"],
    packages=find_packages(exclude=['ez_setup']),
    include_package_data=True,
    test_suite='nose.collector',
    package_data={'pawsapp': ['i18n/*/LC_MESSAGES/*.mo']},
    #message_extractors={'pawsapp': [
    #        ('**.py', 'python', None),
    #        ('templates/**.mako', 'mako', {'input_encoding': 'utf-8'}),
    #        ('public/**', 'ignore', None)]},
    zip_safe=False,
    paster_plugins=['PasteScript', 'Pylons'],
    entry_points="""
    [paste.app_factory]
    main = pawsapp.config.middleware:make_app

    [paste.app_install]
    main = pylons.util:PylonsInstaller

    [paste.paster_command]
    tool-template = scripts.tool:ToolTemplate
    tutorial-template = scripts.tutorial:TutorialTemplate
    """,
)
