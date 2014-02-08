.. Par4All documentation master file, created by
   sphinx-quickstart on Fri Jan 17 15:30:36 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Par4All
=======

Par4All is an automatic parallelizing and optimizing compiler (workbench)
for C and Fortran sequential programs.

The purpose of this source-to-source compiler is to adapt existing
applications to various hardware targets such as multicore systems, high
performance computers and GPUs or some parallel embedded heterogeneous
systems.

It creates new OpenMP, CUDA or OpenCL source codes and thus allows the
original source codes of the application to remain mainly unchanged for
well formed programs.

Par4All is an open source project that merges various open source
developments.

With `Wild Cruncher
<http://www.silkan.com/more-about/#wildcruncher>`_
from `SILKAN <http://www.silkan.com>`_ it is also possible to compile and
parallelize `Scilab <http://www.scilab.org>`_ programs to speed up
computations.

..
  I've not been able to have a better link that this...

.. figure:: images/P4A-video2.png
   :height: 250
   :target: news_and_events.html#video-showing-par4all-features

   A few explanations on Par4All in a :ref:`3mn video <par4all-video>`


Content
-------

.. toctree::
   :maxdepth: 2

   download
   features
   benchmarks
   documentation
   community
   news_and_events


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


..
  # Some Emacs stuff:
  ### Local Variables:
  ### mode: rst,flyspell
  ### ispell-local-dictionary: "american"
  ### End:
