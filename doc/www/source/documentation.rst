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



Publications
------------

To cite Par4ll in a publication about GPUs code generation, the best
references are:

- Mehdi AMINI, Béatrice CREUSILLET, Stéphanie EVEN, Ronan KERYELL, Onil
  GOUBIER, Serge GUELTON, Janice Onanian MCMAHON, François Xavier
  PASQUIER, Grégoire PÉAN and Pierre VILLALON. « **Par4All : From Convex
  Array Regions to Heterogeneous Computing.** » in 2nd International
  Workshop on Polyhedral Compilation Techniques (IMPACT 2012). Paris,
  France, 01/2012.
  http://impact.gforge.inria.fr/impact2012/workshop_IMPACT/amini.pdf
  http://impact.gforge.inria.fr/impact2012/slides/amini.pdf

- Mehdi AMINI. « **Source-to-Source Automatic Program Transformations for
  GPU-like Hardware Accelerators** ». PhD Thesis. Paris, France, 12/2012.
  http://www.cri.ensmp.fr/classement/doc/A-506.pdf


Articles
........

You may find also some other publications on the WWW pages of the authors
and on http://www.cri.ensmp.fr/classement/2014.html

- Mehdi AMINI, Corinne ANCOURT, Béatrice CREUSILLET, François IRIGOIN and
  Ronan KERYELL. Patterns for Parallel Programming on GPUs, Chapter
  « Program Sequentially, Carefully, and Benefit from Compiler Advances for
  Parallel Heterogeneous Computing », pages 151–171. Saxe-Coburg
  Publications, 2014. To appear.

- Mehdi Amini. « Source-to-Source Automatic Program Transformations for
  GPU-like Hardware Accelerators » PhD defense. 12/2012.
  http://www.cri.ensmp.fr/classement/doc/A-506.pdf

- Massimo TORQUATI, Marco VANNESCHI, Mehdi AMINI, Serge GUELTON, Ronan
  KERYELL, Vincent LANORE, François-Xavier PASQUIER, Michel BARRETEAU,
  Rémi BARRÈRE, Claudia-Teodora PETRISOR, Éric LENORMAND, Claudia CANTINI
  and Filippo De STEFANI. « **An innovative compilation tool-chain for
  embedded multi-core architectures.** » in Embedded World
  Conference 2012. Nuremberg, Germany,
  2/2012. http://calvados.di.unipi.it/storage/paper_files/2012_torquati_ew.pdf.

- Nicolas VENTROUX, Tanguy SASSOLAS, Alexandre GUERRE, Béatrice CREUSILLET
  and Ronan KERYELL. « **SESAM/Par4All : A Tool for Joint Exploration of
  MPSoC Architectures and Dynamic Dataflow Code Generation.** » in
  RAPIDO’12, 4th Workshop on : Rapid Simulation and Performance Evaluation
  : Methods and Tools. Paris, France,
  01/ 2012. http://nventrou.free.fr/papers/RAPIDO2012_NV.pdf. Best paper
  award.

- Mehdi AMINI, Béatrice CREUSILLET, Stéphanie EVEN, Ronan KERYELL, Onil
  GOUBIER, Serge GUELTON, Janice Onanian MCMAHON, François Xavier
  PASQUIER, Grégoire PÉAN and Pierre VILLALON. « **Par4All : From Convex
  Array Regions to Heterogeneous Computing.** » in 2nd International
  Workshop on Polyhedral Compilation Techniques (IMPACT 2012). Paris,
  France, 01/2012.
  http://impact.gforge.inria.fr/impact2012/workshop_IMPACT/amini.pdf
  http://impact.gforge.inria.fr/impact2012/slides/amini.pdf

- Mehdi AMINI, Fabien COELHO, François IRIGOIN and Ronan KERYELL. «
  **Static Compilation Analysis for Host-Accelerator Communication
  Optimization.** » in LCPC’2011 : 24th International Workshop on
  Languages and Compilers for Parallel Computing. Fort Collins, Colorado,
  USA, 9/2011. http://www.cri.ensmp.fr/classement/doc/A-476.pdf

- Mehdi AMINI, Corinne ANCOURT, Fabien COELHO, Béatrice CREUSILLET, Serge
  GUELTON, François IRIGOIN, Pierre JOUVELOT, Ronan KERYELL and Pierre
  VILLALON. « **PIPS Is not (just) Polyhedral Software.** » in First
  International Workshop on Polyhedral Compilation Techniques (IMPACT
  2011). Chamonix, France,
  4/2011. http://perso.ens-lyon.fr/christophe.alias/impact2011/impact-09.pdf

- Béatrice Creusillet. « **Automatic Task Generation on the SCMP
  architecture for data flow applications.** » SCALOPES Technical
  Report. 03/2011.
  :download:`download/bc_report2.pdf`


Posters
.......

- Amira MENSI. « **Points-to Analysis for the C Language**
  ». HiPEAC 2012. Best student poster award.
  :download:`download/Amira_Mensi_poster_hipeac2012.pdf`

- Serge GUELTON, Mehdi AMINI, Ronan KERYELL and Béatrice CREUSILLET «
  **PyPS, a programmable pass manager.** » In 24th International Workshop on
  Languages and Compilers for Parallel Computing, Fort Collins, Colorado,
  USA, 9/2011. http://www.cri.ensmp.fr/classement/doc/A-480.png


Presentations
-------------

- 2012/07/04 – `Overview of HPC <http://enstb.org/~keryell/publications/exposes/2012/2012-07-04-Overview_of_HPC-HPC@LR_Montpellier/2012-07-05-HPC-overview-RK-expose.pdf>`_

  Ronan Keryell @ `Linux Cluster Institute 2012
  <https://www.hpc-lr.univ-montp2.fr/lci-2012/programme-129>`_, Université
  Montpellier 2, Centre de compétences, Montpellier, France

- 2012/04/23 – `Par4All: From Sequential Applications to Heterogeneous
  Parallel Computing
  <http://enstb.org/~keryell/publications/exposes/2012/2012-04-23-HPC-GPU_Meetup_CMU/2012-04-23-HPC-GPU_Meetup_CMU-Par4All-expose.pdf>`_

  Ronan Keryell @ `Meetup of HPC & GPU Supercomputing Group of Silicon
  Valley
  <http://www.meetup.com/HPC-GPU-Supercomputing-Group-of-Paris-Meetup/events/43673412/>`_,
  Moffett Field, CA, USA

- 2012/01/25 – `Par4All: Open source parallelization for heterogeneous
  computing OpenCL & more
  <http://enstb.org/~keryell/publications/exposes/2012/2012-01-25-Paris-HPC-GPU-meetup-Par4All/2012-01-25-Paris-HPC-GPU-meetup-Par4All-expose.pdf>`_

  Ronan Keryell @ `HPC & GPU Supercomputing Group of Paris Meetup
  <http://www.meetup.com/HPC-GPU-Supercomputing-Group-of-Paris-Meetup/events/43673412/>`_,
  Paris, France

- 2012/01/24 - `Par4All: Open source parallelization for heterogeneous
  computing OpenCL & more
  <http://enstb.org/~keryell/publications/exposes/2012/2012-01-24-HiPEAC-OpenGPU-Par4All/Par4All-HiPEAC-OpenGPU-expose.pdf>`_

  Ronan Keryell @ `HiPEAC 2012 <http://www.hipeac.net/hipeac2012>`_ /
  `Workshop OpenGPU
  <http://opengpu.net/index.php?option=com_content&view=article&id=157&Itemid=144>`_,
  Paris, France

- 2009/10/01 – `Par4All: Auto-Parallelizing C and Fortran for the CUDA
  Architecture
  <http://download.par4all.org/doc/presentations/2009/nVidia-GPU_Technology_Conference-2009/Par4All_Cuda-RK-expose.pdf>`_

 A presentation of Par4All by Ronan Keryell at the `nVidia GPU Technology
 Conference
 <http://www.nvidia.com/object/gpu_technology_conference.html>`_ in San
 José, Ca. `Printer friendly version
 <http://download.par4all.org/doc/presentations/2009/nVidia-GPU_Technology_Conference-2009/Par4All_Cuda-RK-copie.pdf>`_

- 2009/07/01 – `GPU & Open Source
  <http://download.par4all.org/doc/presentations/2009/Forum_Ter@tec_2009-OpenGPU/Ter@tec_2009-OpenGPU-RK-expose.pdf>`_

  Presentation by Ronan Keryell from HPC Project at the OpenGPU session at
  the `Ter@tec Forum 2009 <http://www.teratec.eu/forum>`_, 2009/07/01. `Printer friendly version <http://download.par4all.org/doc/presentations/2009/Forum_Ter@tec_2009-OpenGPU/Ter@tec_2009-OpenGPU-RK-copie.pdf>`_

- ESWEEK 2009 Panel: `compilers for embedded systems
  <http://download.par4all.org/doc/presentations/2009/ESWEEK-2009-10-12/ESWEEK-Panel-RK-expose.pdf>`_

  The contribution of Ronan Keryell to the panel at `ESWEEK
  <http://esweek09.inrialpes.fr>`_, 10/14/2009 in Grenoble. `Printer
  friendly version
  <http://download.par4all.org/doc/presentations/2009/ESWEEK-2009-10-12/ESWEEK-Panel-RK-copie.pdf>`_


..
  # Some Emacs stuff:
  ### Local Variables:
  ### mode: rst,flyspell
  ### ispell-local-dictionary: "american"
  ### End:
