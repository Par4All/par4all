/** @file

    API of Par4All C for accelerator

    @mainpage

    The Par4All API leverages the code to program an hardware accelerator
    with some helper functions.

    It allows simpler and more portable code to be generated with PIPS,
    but it can also be used to ease heterogeneous parallel manual
    programming.

    Should target Ter\@pix and SCALOPES accelerators, CUDA and OpenCL.

    Funded by the FREIA (French ANR), TransMedi\@ (French Pôle de
    Compétitivité Images and Network) and SCALOPES (Artemis European
    Project project)

    "mailto:Ronan.Keryell@hpc-project.com"
*/

/* License BSD */

#ifndef P4A_ACCEL_H
#define P4A_ACCEL_H

/** Note that in CUDA and OpenCL there is 3 dimensions max: */
enum { P4A_vp_dim_max = 3 };


#ifdef P4A_ACCEL_CUDA
#include <p4a_accel-CUDA.h>
#else
#ifdef P4A_ACCEL_OPENMP
#include <p4a_accel-OpenMP.h>
#else
#include <p4a_accel-.h>
#endif
#endif

/** A macro to enable or skip debug instructions

    Just define P4A_DEBUG to have debug information at runtime

    @param debug_stuff is some text that is included texto if P4A_DEBUG is
    defined
*/
#ifdef P4A_DEBUG
#define P4A_SKIP_DEBUG(debug_stuff) debug_stuff
#else
#define P4A_SKIP_DEBUG(debug_stuff)
#endif


/** Output a debug message à la printf */
#define P4A_DUMP_MESSAGE(...)			\
  fprintf(stderr, " P4A: " __VA_ARGS__)


/** A macro to compute an minimum value of 2 values

    Since it is a macro, beware of side effects...
*/
#define P4A_MIN(a, b) ((a > b) ? b : a)

#endif //P4A_ACCEL_H
