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



Background
----------

In 1988 the CRI lab (*Centre de recherche Informatique*) of Mines
ParisTech began developing PIPS (in French: :emphasis:`Parallélisation
Interprocédurale de Programme Scientifiques`) as a source to source
compiler.

In 2009, HPC Project (now called SILKAN) with the CRI announced a new open
source platform for automatic parallelisation of computer programs:
**Par4All**.

Internally, Par4All is currently composed of different components:

- the `PIPS <http://pips4u.org>`_ source-to-source compiler that began at
  `MINES ParisTech <http://cri.mines-paristech.fr>`_ in 1988 and is
  currently developed also in many other places: `SILKAN
  <http://www.silkan.com>`_, `Institut TÉLÉCOM/TÉLÉCOM Bretagne
  <http://departements.telecom-bretagne.eu/info>`_, `IT SudParis
  <http://inf.telecom-sudparis.eu>`_, `RPI (Rensselaer Polytechnic
  Institute) <http://www.cs.rpi.edu>`_.

- the `PolyLib <http://icps.u-strasbg.fr/polylib/>`_ used by PIPS,

- GCC/GFC for the Fortran95 parser,

- and of course own tools and scripts to make all these components and the
  global infrastructure usable.

.. image:: images/Mines-paris-tech.jpg

`CRI MINES ParisTech <http://www.cri.mines-paristech.fr>`_ : The « Centre
de Recherche en Informatique, Mathématiques et systèmes, MINES ParisTech »
is the Center for research in computing, mathematics and systems from the
École des Mines belonging to the ParisTech group. The CRI studies
languages used in computer science such as programming, data description,
or query languages. The CRI develops semantic analysis and automatic
transformation of these languages to answer industrial needs (performance,
development cost and time-to-market) as well as administrative and
societal needs (coherent data sharing, data normalization, access to data
and heritage protection).


.. image:: images/silkan-logo1_RVB.jpg

`SILKAN <http://silkan.com>`_ (formely called HPC Project) delivers
cost-effective application-in-a-box solutions for demanding users who
require intense computational power. The company also assists its
customers with a set of services allowing to leverage the power of
latest-generation processors. You can find more information about SILKAN
on our corporate web site or on our page dedicated to our
Application-in-a-box solution, Wild Systems. SILKAN is the result of the
merge of HPC Project and Arion Entreprise.


..
  # Some Emacs stuff:
  ### Local Variables:
  ### mode: rst,flyspell
  ### ispell-local-dictionary: "american"
  ### End:
