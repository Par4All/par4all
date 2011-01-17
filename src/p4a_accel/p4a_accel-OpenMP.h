/** @file

    API of Par4All C to OpenMP

    This file is an implementation of the OpenMP Par4All API.

    Funded by the FREIA (French ANR), TransMedi\@ (French Pôle de
    Compétitivité Images and Network) and SCALOPES (Artemis European
    Project project), with the support of the MODENA project (French
    Pôle de Compétitivité Mer Bretagne)

    "mailto:Stephanie.Even@enstb.org"
    "mailto:Ronan.Keryell@hpc-project.com"

    This work is done under MIT license.

    
*/

//Do not include twice this file:
#ifndef P4A_ACCEL_OPENMP_H
#define P4A_ACCEL_OPENMP_H

#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

// Includes the kernel and variable qualifiers.
//  Some are used when launching the kernel.

#include "p4a_accel_wrapper-OpenMP.h"

/** @addtogroup P4A_time_measure Time measurement

    In OpenMP, the time is measured via the call to the
    gettimeofday().  In OpenMP, two timeval struct are used to record the
    absolute time, once when calling the P4A_accel_timer_start, once
    when calling the P4A_accel_timer_stop_and_float_measure(). The
    elapsed time is the difference between the two recorded times,
    stored in p4a_time_begin and p4a_time_end.

    @{
*/

/** In OpenMP, the absolute time recorded when the timer is started
 */
struct timeval p4a_time_begin; 

/** In OpenMP, the absolute time recorded when the timer ends
 */
struct timeval p4a_time_end;

/** Start a timer on the host for OpenMP.

    Records the p4a_time_begin absolute time.
*/
#ifdef P4A_PROFILING
#define P4A_accel_timer_start gettimeofday(&p4a_time_begin, NULL)
#else
#define P4A_accel_timer_start 
#endif
/** 
    @} 
*/

/** @defgroup P4A_init_release Begin and end actions 

    @{
*/

/** Associate the program to the accelerator as OpenMP

    Initialized the use of the hardware accelerator: nothing to do since
    there is no accelerator, only OpenMP with local processors...
*/
#define P4A_init_accel


/** Release the hardware accelerator as OpenMP

    Nothing to do
 */
#define P4A_release_accel

/** @} */

/** @defgroup P4A_kernel Accelerator kernel call

    An API for full kernel call control. 

    The API call scheme is driven by CUDA and OpenCl API. It uses three
    arguments types:
    -# the kernel wrapper name. The kernel call with its grid dimension 
    (the context) is constructed in the P4A_call_accel_kernel_context.
    -# the 1D, 2D or 3D dimensions of the computation grid. The grid 
    descriptors are constructed using the 
    P4A_create_1d_thread_descriptors(P4A_n_iter_0) or 
    P4A_create_2d_thread_descriptors(P4A_n_iter_0,P4A_n_iter_1).
    -# the parameters list is passed via the P4A_call_accel_kernel_parameters.

    In addition, the kernel are externalized in specific files. This
    constrains is imposed by OpenCL. The prototype of the kernel must
    be declared in the main file where it is launched. A
    P4A_wrapper_proto was also created that must be declared in the
    main file.

    @{
*/

/** @defgroup P4A_kernel_protoizer Kernel protoizer.

    Kernel declaration in the main file since the kernel is externalized in specific files; 

    @{

    In OpenMP, the prototype of the kernel is the classical
    signature in C. P4A_accel_kernel_wrapper is a kernel qualifier defined in
    p4a_accel_wrapper-OpenMP.h.
*/
#define P4A_wrapper_proto(kernel, ...)		\
  P4A_accel_kernel_wrapper kernel(__VA_ARGS__)

/** 
    @}
*/


/** @defgroup P4A_kernel_context Kernel contextualization.

    Call the kernel function associated with a specific context.

    @{

    No specific kernel call over a grid in OpenMP.
    In the OpenMP implementation, the P4A_call_accel_kernel that aims at
    forming the kernel context (call over a grid descriptor) is not used.
    This is not yet implemented. Ask for it if you really want it (that is
    you need the dim3 grids of blocks and blocks of threads). Right now
    only the functions useful for codegeneration from PIPS has been
    implemented.
*/

/** In OpenMP, no need for context.
*/
#define P4A_call_accel_kernel_context(kernel, ...) \
  error "P4A_call_accel_kernel_context not yet implemented in OpenMP"

/** 
    @}
*/


/** @defgroup P4A_kernel_parameters Parameters invocation
  
    Construct the parameter list for the kernel.

    @{

    Presently not implemented in OpenMP.
*/

/** In OpenMP, the kernel call via a context is not used. Also, this
    specific call of parameters is not used.
    
    This is not yet implemented.
*/
#define P4A_call_accel_kernel_parameters(...) \
  error "P4A_call_accel_kernel_parameters not yet implemented in OpenMP"

/** 
    @}
*/

/** @defgroup P4A_grid_descriptors Constructors for the the grid descriptor

    Only used in CUDA and OpenCL.
    @{
*/

/** Allocate the descriptors for a linear set of thread with a simple
    strip-mining with an OpenMP implementation.
*/
#define P4A_create_1d_thread_descriptors(block_descriptor_name,		\
					 grid_descriptor_name,		\
					 size)				\
  error "P4A_create_1d_thread_descriptors not yet implemented in OpenMP"


/** Allocate the descriptors for a 2D set of thread with a simple
    strip-mining in each dimension with an OpenMP implementation.
*/
#define P4A_create_2d_thread_descriptors(block_descriptor_name,		\
					 grid_descriptor_name,		\
					 n_x_iter, n_y_iter)		\
  error "P4A_create_2d_thread_descriptors not yet implemented in OpenMP"


/** Dump a CUDA dim3 descriptor with an introduction message with an
    OpenMP implementation.
*/
#define P4A_dump_descriptor(message, descriptor_name) \
  error "P4A_dump_descriptor not yet implemented in OpenMP"


/** Dump a CUDA dim3 block descriptor with an OpenMP implementation.
*/
#define P4A_dump_block_descriptor(descriptor_name) \
  error "P4A_dump_block_descriptort not yet implemented in OpenMP"


/** Dump a CUDA dim3 grid descriptor with an OpenMP implementation.
*/
#define P4A_dump_grid_descriptor(descriptor_name) \
  error "P4A_dump_grid_descriptor not yet implemented in OpenMP"


/**
    @}
 */

/** @defgroup P4A_kernel_call Kernel call constructor

    The kernel call itself.

    For simple usage
    @see P4A_call_accel_kernel_1d, 
    @see P4A_call_accel_kernel_2d, 
    @see P4A_call_accel_kernel_3d

    @{
*/


/** In OpenMP, no need for context.
 */
#define P4A_call_accel_kernel(context, parameters) \
  error "P4A_call_accel_kernel not yet implemented in OpenMP"



/** Call a kernel in a 1-dimension parallel loop with OpenMP emulation.

    Translate it into an OpenMP parallel loop

    @param[in] kernel to call

    @param[in] P4A_n_iter_0 is the number of iterations

    @param ... the following parameters are given to the kernel
*/
#define P4A_call_accel_kernel_1d(kernel, P4A_n_iter_0, ...)		\
  P4A_skip_debug(P4A_dump_location();)					\
  P4A_skip_debug(P4A_dump_message("P4A_call_accel_kernel_1d(%d) of \"%s\"\n", P4A_n_iter_0, #kernel);)	\
  _Pragma("omp parallel for")						\
  for(int P4A_index_0 = 0; P4A_index_0 < P4A_n_iter_0; P4A_index_0++) {	\
    P4A_vp_0 = P4A_index_0;						\
    P4A_vp_1 = 0;							\
    P4A_vp_2 = 0;							\
    kernel(__VA_ARGS__);						\
  }


/** Call a kernel in a 2-dimension parallel loop with OpenMP emulation.

    Translate it into an OpenMP parallel loop-nest

    @param[in] kernel to call

    @param[in] P4A_n_iter_0 is the number of iterations in the first dimension

    @param[in] P4A_n_iter_1 is the number of iterations in the second dimension

    @param ... following parameters are given to the kernel
*/
#define P4A_call_accel_kernel_2d(kernel, P4A_n_iter_0, P4A_n_iter_1, ...) \
  P4A_skip_debug(P4A_dump_location();)					\
  P4A_skip_debug(P4A_dump_message("P4A_call_accel_kernel_2d(%d,%d) of \"%s\"\n", P4A_n_iter_0, P4A_n_iter_1, #kernel);)	\
  _Pragma("omp parallel for")						\
  for(int P4A_index_0 = 0; P4A_index_0 < P4A_n_iter_0; P4A_index_0++) {	\
    for(int P4A_index_1 = 0; P4A_index_1 < P4A_n_iter_1; P4A_index_1++) { \
      P4A_vp_0 = P4A_index_0;						\
      P4A_vp_1 = P4A_index_1;						\
      P4A_vp_2 = 0;							\
      P4A_skip_debug(P4A_dump_message("%d %d\n", P4A_vp_0, P4A_vp_1));	\
      kernel(__VA_ARGS__);						\
    }									\
  }


/** Call a kernel in a 3-dimension parallel loop with OpenMP emulation.

    Translate it into an OpenMP parallel loop-nest

    @param[in] kernel to call

    @param[in] P4A_n_iter_0 is the number of iterations in the first dimension

    @param[in] P4A_n_iter_1 is the number of iterations in the second dimension

    @param[in] P4A_n_iter_2 is the number of iterations in the third dimension

    @param ... following parameters are given to the kernel
*/
#define P4A_call_accel_kernel_3d(kernel, P4A_n_iter_0, P4A_n_iter_1, P4A_n_iter_2, ...) \
  P4A_skip_debug(P4A_dump_location();)					\
  P4A_skip_debug(P4A_dump_message("P4A_call_accel_kernel_3d(%d,%d,%d) of \"%s\"\n", P4A_n_iter_0, P4A_n_iter_1, P4A_n_iter_2, #kernel);)	\
  _Pragma("omp parallel for")						\
  for(int P4A_index_0 = 0; P4A_index_0 < P4A_n_iter_0; P4A_index_0++) {	\
    for(int P4A_index_1 = 0; P4A_index_1 < P4A_n_iter_1; P4A_index_1++) { \
      for(int P4A_index_2 = 0; P4A_index_2 < P4A_n_iter_2; P4A_index_2++) { \
        P4A_vp_0 = P4A_index_0;						\
        P4A_vp_1 = P4A_index_1;						\
        P4A_vp_2 = P4A_index_2;						\
        kernel(__VA_ARGS__);						\
      }									\
    }									\
  }

/** @} */
/** @} */

#endif
