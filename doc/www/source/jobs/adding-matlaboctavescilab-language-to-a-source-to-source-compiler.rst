Adding MatLab/Octave/SciLab language to a source-to-source compiler
===================================================================

Internship for a MSc or engineer student.

**Keywords:** MatLab, Octave, SciLab, compilation, parallelization,
program transformation, abstract interpretation, type reconstruction

Dynamic languages oriented toward mathematical simulations such as
MatLab/Octave/SciLab are quite fashionable for the rapid prototyping of
complex applications. Unfortunately, their execution is not very quite
efficient and often to slow compared to highly optimized coding in a
statically compiled imperative language (C, Fortran, C++).

Current computing architectures are all evolving toward parallel
architectures with always more processors for power-efficiency reasons:
processors with 8+ cores, GPU for graphics and computations with 1600
cores (for example the old 2.7-TFLOPS AMD/ATI HD 5870)... This means that
the stress is put on the programmer to parallelize her application to such
an extend to fit the target application. Unfortunately, this is quite more
difficult than plain sequential programming and there is no direct path
for languages such as MatLab/Octave/SciLab.

The parallel execution can be hidden in specialized libraries, but in this
case only this part of the execution is optimized, or the parallelism can
be hidden in some objects (for example a matrix on a parallel accelerator)
but in this case the programmer must change the source code to use this
new kinds of objects, and of course only some computations on these
objects will be accelerated, say, on a GPU.

Since there exist compilers that are able to parallelize some languages
like C or Fortran, it is appealing to be able to adapt these compiler
technologies for languages such as MatLab and Octave. Unfortunately a C
compiler cannot deal with such a language, and even if we add a dedicated
parser to it to deal with a new language, major improvements will be
needed in the abstract syntax tree and the semantics analysis to deal with
all these new dynamic constructs.

In this internship, we want to try a new direction: developing a new tool
using the MatLab/Octave free parser and generate equivalent raw C code
with some calls to intrinsics functions to add the needed expressiveness
lacking to a pure C translation. This translated C will be analyzed by the
PIPS source-to-source compiler to generate an efficient equivalent
parallel C code. For this, many improvement will be needed to deal with
the new intrinsics functions and many new transformation phases will be
needed to optimized the code before final generation.

The translated code will be profiled against multicore and GPU executions
for performance comparison.

This internship can be followed by a full-time job or a PhD thesis.

Advisor: Thierry.Porcher (at)hpc-project dot com with Mines ParisTech/CRI

Montpellier, France.

..
  # Some Emacs stuff:
  ### Local Variables:
  ### mode: rst,flyspell
  ### ispell-local-dictionary: "american"
  ### End:
