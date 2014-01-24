Coupling a pattern-matching language to a source-to-source compiler
===================================================================

Internship for a MSc or engineer student

**Keywords**: pattern-matching language, compilation, parallelization,
 program transformation, optimization

Current computing architectures are all evolving toward parallel
architectures with always more processors for power-efficiency reasons:
processors with 8+ cores, GPU for graphics and computations with 1600
cores (for example the old 2.7-TFLOPS AMD/ATI HD 5870)… This means that
the stress is put on the programmer to parallelize her application to such
an extend to fit the target application. Unfortunately, this is quite more
difficult than plain sequential programming.

To help the programmer job, many tools are developed: new parallel
languages, new parallel libraries, new autoparallelizer compilers for
existing languages, etc.

In this internship, we focus on autoparallelizer compilers with the
ability to find out in the programmer source code some constructions that
can efficiently be executed on parallel architectures with specific
instructions (SSE intrinsics for example) or libraries (optimized linear
algebra function such as a matrix multiplication).

The pattern matching is an old research domain we don’t want to reinvent
and the aim of the internship is to find the best pattern-matching tool
(http://tom.loria.fr seems a good candidate…) and language suitable to our
domain and to couple it with ROSE Compiler to have a powerful
source-to-source compiler with advanced algorithm recognition system.

Advisor: Ronan Keryell rk(at)hpc-project(dot)com

http://par4all.org

Location: HPC Project http://www.hpc-project.com which is a fast-growing
start-up with 35 people at Meudon (92) and Montpellier (34), France and
Santa Clara (USA).

..
  # Some Emacs stuff:
  ### Local Variables:
  ### mode: rst,flyspell
  ### ispell-local-dictionary: "american"
  ### End:
