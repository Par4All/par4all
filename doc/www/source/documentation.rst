Documentation
=============

Users guide
-----------

- ``p4a`` is the basic command line interface used to do simple things
  with Par4All: parallelizing programs, compiling with backend. The
  Par4All user manual with a description of ``p4a`` is available in `PDF
  <http://download.par4all.org/doc/par4all_user_guide/par4all_user_guide.pdf>`_
  and `HTML
  <http://download.par4all.org/doc/par4all_user_guide/par4all_user_guide.htdoc>`_.

- Installation guide in `PDF
  <http://download.par4all.org/doc/installation_guide/par4all_installation_guide.pdf>`_
  and `HTML
  <http://download.par4all.org/doc/installation_guide/par4all_installation_guide.htdoc>`_.

- Some coding rules to write applications with good parallelization
  properties for Par4All in `PDF
  <http://download.par4all.org/doc/p4a_coding_rules/p4a_coding_rules.pdf>`_
  and `HTML
  <http://download.par4all.org/doc/p4a_coding_rules/p4a_coding_rules.htdoc>`_.
  There is no magic powder, parallelization can work only on well written
  programs. There is a more recent and general technical report that can
  be found at http://www.cri.ensmp.fr/classement/doc/A-503.pdf

- Slides presenting ``p4a`` command line interface, generated
  automatically with the help of ``p4a`` (``p4a -h``) and explaining the
  ``p4a`` capabilities.

- Par4All primer with ``tpips``: Introductive slide presentation for
  advanced users of the ``tpips`` command line interface of PIPS present
  in Par4All.


Developers Guide
----------------

- The developer guide describes the internal organization of Par4All and
  its construction and is available in `PDF
  <http://download.par4all.org/doc/developer_guide/par4all_developer_guide.pdf>`_
  and `HTML
  <http://download.par4all.org/doc/developer_guide/par4all_developer_guide.htdoc>`_.

- The **Par4All Accel Runtime** is the adaptation layer used by the ParAll
  compiler backend to address heterogeneous accelerators (GPU for
  example). It can also be directly used by programmers that want to
  address low-level programming while remaining more abstract from the
  architectural point of view. `The Doxygen documentation is here
  <http://download.par4all.org/doc/Par4All_Accel_runtime/graph>`_


..
  # Some Emacs stuff:
  ### Local Variables:
  ### mode: rst,flyspell
  ### ispell-local-dictionary: "american"
  ### End:
