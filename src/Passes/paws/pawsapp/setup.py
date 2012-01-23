import os

from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))
README = open(os.path.join(here, 'README.txt')).read()
CHANGES = open(os.path.join(here, 'CHANGES.txt')).read()

requires = [
    'pyramid',
    'waitress',
    'webhelpers',
    'pyro',
    'PIL',
    ]

setup(name='pawsapp',
      version='1.0',
      description='PAWS, a Web Interface for PIPS',
      long_description=README + '\n\n' +  CHANGES,
      classifiers=[
        "Programming Language :: Python",
        "Framework :: Pylons",
        "Topic :: Internet :: WWW/HTTP",
        "Topic :: Internet :: WWW/HTTP :: WSGI :: Application",
        ],
      author='Maria SZYMCZAK',
      author_email='',
      url='http://paws.pips4u.org',
      keywords='web pyramid pylons pip computing compilation',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False,
      install_requires=requires,
      tests_require=requires,
      test_suite="pawsapp",
      entry_points = """\
      [paste.app_factory]
      main = pawsapp:main
      """,
      )

