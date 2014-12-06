Download - Par4All
==================

Because of current limitations in PIPS there are some specific bugs on
recent 32-bit *x*\ 86 Linux target so please use a 64-bit platform to run
Par4All.

`Release note <https://github.com/Par4All/par4all/blob/p4a/LICENSE.txt>`_

`Changelog <https://github.com/Par4All/par4all/blob/p4a/src/simple_tools/DEBIAN/changelog>`_

`License <https://github.com/Par4All/par4all/blob/p4a/LICENSE.txt>`_

If you need to run Par4All on unsupported system, please run Par4All
inside a virtual machine running a supported OS.

**Warning:** since this project is no longer supported by SILKAN, if you
want to use the latest version of Par4All, the only way from the following
methods is to start from the source from
https://github.com/Par4All/par4all
described in section `Distributed version control system`_.

Binary distribution
-------------------

Right now Par4All is only supported in binary form on Debian and Ubuntu
64-bit *x*\ 86.


Repository
..........

The best way if you are on GNU/Linux Debian or Ubuntu is to use our
package repository. This way, when a new version is out, your classical
package manager can automatically install it.

To use our package repository, pick one of the following lines, and add it
graphically with the **Update Manager** (``Settings``/``Third-Party
Software``) or append it with a text editor to your
``/etc/apt/sources.list`` file.

If you are using Ubuntu: ::

  deb http://download.par4all.org/apt/ubuntu releases main
  # --OR--
  deb http://download.par4all.org/apt/ubuntu development main

or if you are running Debian: ::

  deb http://download.par4all.org/apt/debian releases main
  # --OR--
  deb http://download.par4all.org/apt/debian development main

So you need to choose between releases or development
versions. Development packages are generated often, may be unstable, and
are best suited if you want to track more closely the Par4All development.

Once this is done, run your favorite graphics package tool (synaptic...)
or:

.. code:: bash

  sudo apt-get update
  sudo apt-get install par4all

In any case, you will then need to source one of the following shell
scripts which set up the environment variables for proper Par4All
execution:

- if you use ``bash``, ``sh``, ``dash``, etc.:

  .. code:: bash

    source /usr/local/par4all/etc/par4all-rc.sh

- if you use ``csh``, ``tcsh``, etc.:

  .. code:: bash

    source /usr/local/par4all/etc/par4all-rc.csh

For more information, look at the `documentation`_ section.


Package
.......


A less automatic way on Debian or Ubuntu is to install a Par4All ``.deb``
package manually.

For development versions, according to your OS and architecture, download
a package from: ::

    http://download.par4all.org/development/debian/i686
    http://download.par4all.org/development/debian/x86_64
    http://download.par4all.org/development/ubuntu/i686
    http://download.par4all.org/development/ubuntu/x86_64

For release versions, according to your OS and architecture, download a
package from: ::

    http://download.par4all.org/releases/debian/i686
    http://download.par4all.org/releases/debian/x86_64
    http://download.par4all.org/releases/ubuntu/i686
    http://download.par4all.org/releases/ubuntu/x86_64

You can then install the package with:

.. code:: bash

  sudo gdebi <the_package>.deb

``sudo dpkg -i <the_package>.deb`` would also work but does not
automatically install dependencies you should install later.

In any case, you will then need to source one of the following shell
scripts which set up the environment variables for proper Par4All
execution:

- if you use ``bash``, ``sh``, ``dash``, etc.:

  .. code:: bash

    source /usr/local/par4all/etc/par4all-rc.sh

- if you use ``csh``, ``tcsh``, etc.:

  .. code:: bash

    source /usr/local/par4all/etc/par4all-rc.csh

For more information, look at the `documentation`_ section.


Manual tar.gz binary installation
.................................

An even less automatic way is to use a ``.tar.gz`` tar-ball file. It
contains the binaries as built on a stable Ubuntu or unstable Debian
distribution. It should work on any GNU/Linux distribution with the
following libraries installed: (a fairly recent) ``libc.so.6``,
``libncurses.so.5``, ``libreadline.so.6``, etc. and Python 2.7. We chose
this Python version because it is recent enough to provide nice features
for Par4All and not too recent to be absent from most Linux distributions…
Look at the Par4All organization documentation to have the list of some
needed packages.

Once you have downloaded one of these ``.tar.gz`` packages from
http://download.par4all.org\ , extract it with the following command: ::

  tar xvzf <the_package>.tar.gz

It will create a directory named par4all. Move this directory to its final
location, for example with: ::

  sudo mv par4all /usr/local

In any case, you will then need to source one of the following shell
scripts which set up the environment variables for proper Par4All
execution:

- if you use ``bash``, ``sh``, ``dash``, etc.:

  .. code:: bash

    source /usr/local/par4all/etc/par4all-rc.sh

- if you use ``csh``, ``tcsh``, etc.:

  .. code:: bash

    source /usr/local/par4all/etc/par4all-rc.csh

For more information, look at the `documentation`_ section.


Previous releases
-----------------

Older releases of Par4All packages are available on
http://download.par4all.org/releases


Installing from the sources
---------------------------

This is not the preferred way to work, but it can be useful for people who
cannot use a precompiled version and do not want to bother with ``git``.

First get a source tar-ball in the following directories (Ubuntu or Debian
do not matter here): ::

  http://download.par4all.org/development
  http://download.par4all.org/releases

Pick up a file which name ends with ``_src.tar.gz``. You can decompress it
with a ``tar zxvf``.

Then refer to the infrastructure documentation on how to compile with
``p4a_setup.py`` after having installed the required packages.


Distributed version control system
----------------------------------

Since this project is no longer supported by SILKAN, if you want to use
the latest version of Par4All, the only way from the following methods is
to start from the source from https://github.com/Par4All/par4all

You can also access to the latest Par4All source code and contribute using
``git``:

=============================================  =========================================
``git`` source viewer @ GitHub (most recent)    https://github.com/Par4All/par4all.git
``git`` source viewer @ SILKAN (old)            https://git.silkan.com/cgit/par4all
GitHub ``ssh`` access (most recent)             git@github.com:Par4All/par4all.git
Anonymous ``git`` access @ SILKAN (old)         git://git.par4all.org/par4all
Commit ``git`` access @ SILKAN (old)            ssh://git.silkan.com/git/par4all.git
=============================================  =========================================


To compile from the sources or from ``git``, have a look at the Par4All
organization `Documentation <documentation>`_

The ``git`` repository on GitHub is a cleaned-up version without some big
files not allowed on GitHub.

..
  # Some Emacs stuff:
  ### Local Variables:
  ### mode: rst,flyspell
  ### ispell-local-dictionary: "american"
  ### End:
