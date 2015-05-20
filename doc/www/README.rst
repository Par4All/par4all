The WWW site is generated with Sphinx from ReST static files.

http://par4all.github.io

It is to be compiled and published on GitHub Pages by using the
``gh-pages`` branch through the ``src/dev/publish_GitHub_Pages`` script.

For GitHub, the publication goes through the
``src/dev/publish_par4all_without_big_file`` script that remove some big
files too big to be accepted by GitHub... ``:-(``

You can build the web site locally with: ::

  make html

and publish it on GitHub when the ``p4a-own`` branch is committed with: ::

  make publish

The file ``build/html/CNAME`` controls the fact that this site
was hosted as http://www.par4all.org by GitHub Pages.
@todo: move this file to the source part.

@todo the version number is in ``source/conf.py`` but should use directly
Par4All ``VERSION`` file.

It is now http://par4all.github.io as it is pushed in a repository
named ``par4all.github.io``.
