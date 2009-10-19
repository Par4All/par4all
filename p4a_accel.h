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

    "mailto:Stephanie.Even@enstb.org"
    "mailto:Ronan.Keryell@hpc-project.com"
*/

/* License BSD */

#ifndef P4A_ACCEL_H
#define P4A_ACCEL_H

/** Note that in CUDA and OpenCL there is 3 dimensions max: */
enum { P4A_vp_dim_max = 3 };

extern double P4A_accel_timer_stop_and_float_measure();

#if defined(P4A_ACCEL_CUDA) && defined(P4A_ACCEL_OPENMP)
#error "You cannot have both P4A_ACCEL_CUDA and P4A_ACCEL_OPENMP defined, yet"
#endif

#ifdef P4A_ACCEL_CUDA
#include <p4a_accel-CUDA.h>
#else
#ifdef P4A_ACCEL_OPENMP
#include <p4a_accel-OpenMP.h>
#else
#error "You have to define either P4A_ACCEL_CUDA or P4A_ACCEL_OPENMP"
#endif
#endif

/** A macro to enable or skip debug instructions

    Just define P4A_DEBUG to have debug information at runtime

    @param debug_stuff is some text that is included texto if P4A_DEBUG is
    defined
*/
#ifdef P4A_DEBUG
#define P4A_skip_debug(debug_stuff) debug_stuff
#else
#define P4A_skip_debug(debug_stuff)
#endif


/** Output a debug message à la printf */
#define P4A_dump_message(...)			\
  fprintf(stderr, " P4A: " __VA_ARGS__)


/** A macro to compute an minimum value of 2 values

    Since it is a macro, beware of side effects...
*/
#define P4A_min(a, b) ((a > b) ? b : a)

#endif //P4A_ACCEL_H
