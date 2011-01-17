/** @file

    API of Par4All C to OpenMP, C for CUDA and OpenCL. 

    This file drives the inclusion of the files containing the API for
    the kernel wrapper and kernel declarations. The API of Par4All
    defines a single interface between OpenMP, C for CUDA and
    OpenCL. On the one hand, OpenCL requires that the kernel is loaded
    as a string. To do this more conveniently and in a way compatible
    with OpenMP and C for CUDA, Par4All has choosen to externalized
    the kernels in separate files. On the second hand, the OpenCL
    kernel file does not include files and thus, must be autonomous
    concerning the MACROS declarations. As including the whole
    p4a_accel.h was not possible, because standard header were
    concerned, the only way was to isolate the MACROS specific to the
    kernel and kernel wrapper declarations.  This header is included
    in each kernel file. In OpenMP and C for CUDA mode it is treated
    as usual. In OpenCL, the preprocessor is used to generates the .cl
    version with explicit declaration of the present macros.

    Funded by the FREIA (French ANR), TransMedi\@ (French Pôle de
    Compétitivité Images and Network) and SCALOPES (Artemis European
    Project project), with the support of the MODENA project (French
    Pôle de Compétitivité Mer Bretagne)

    "mailto:Stephanie.Even@enstb.org"
    "mailto:Ronan.Keryell@hpc-project.com"

    This work is done under MIT license.
*/

#ifndef P4A_ACCEL_WRAPPER_H
#define P4A_ACCEL_WRAPPER_H

#ifdef P4A_ACCEL_CUDA
#include <p4a_accel_wrapper-CUDA.h>
#endif

#ifdef P4A_ACCEL_OPENMP
#include <p4a_accel_wrapper-OpenMP.h>
#endif

#ifdef P4A_ACCEL_OPENCL
#include <p4a_accel_wrapper-OpenCL.h>
#endif

#endif //P4A_ACCEL_WRAPPER_H
