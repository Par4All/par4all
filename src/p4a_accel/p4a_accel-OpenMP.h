/** @file

    API of Par4All C to OpenMP

    This file is an implementation of the C to CUDA Par4All API.

    Funded by the FREIA (French ANR), TransMedi\@ (French Pôle de
    Compétitivité Images and Network) and SCALOPES (Artemis European
    Project project)

    "mailto:Stephanie.Even@enstb.org"
    "mailto:Ronan.Keryell@hpc-project.com"

    This work is done under MIT license.
*/

/* Do not include twice this file: */
#ifndef P4A_ACCEL_OPENMP_H
#define P4A_ACCEL_OPENMP_H

#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <p4a_stacksize_test.h>

// no qualifier for devices in openmp emulation mode
#define P4A_DEVICE 

extern struct timeval p4a_time_begin, p4a_time_end;

/** This is a global variable used to simulate P4A virtual processor
    coordinates in OpenMP because we need to pass a local variable to a
    function without passing it in the arguments.

    Use thead local storage (TLS) to have it local to each OpenMP thread.
 */
extern __thread int P4A_vp_coordinate[P4A_vp_dim_max];


/** @defgroup P4A_OpenMP_time_measure Time execution measurement

    @{
*/

/** Start a timer on the host for OpenMP implementation */
#define P4A_accel_timer_start gettimeofday(&p4a_time_begin, NULL)
/** @} */

/** @defgroup P4A_init_OpenMP Initialization of P4A C to OpenMP

    @{
*/

/** Associate the program to the accelerator as OpenMP

    Initialized the use of the hardware accelerator: nothing to do since
    there is no accelerator, only OpenMP with local processors...
*/
#define P4A_init_accel p4a_init_openmp_accel();

void p4a_init_openmp_accel ();

/** Release the hardware accelerator as OpenMP

    Nothing to do
 */
#define P4A_release_accel

/** @} */


/** A declaration attribute of a hardware-accelerated kernel in OpenMP

    Nothing by default for OpenMP since it is normal C
*/
#define P4A_accel_kernel void


/** A declaration attribute of a hardware-accelerated kernel called from
    the host in OpenMP

    Nothing by default since it is homogeneous programming model
*/
#define P4A_accel_kernel_wrapper void


/** Get the coordinate of the virtual processor in X (first) dimension in
    OpenMP emulation. It corresponds to the deepest loop. */
#define P4A_vp_0 P4A_vp_coordinate[0]


/** Get the coordinate of the virtual processor in Y (second) dimension in
    OpenMP emulation */
#define P4A_vp_1 P4A_vp_coordinate[1]


/** Get the coordinate of the virtual processor in Z (second) dimension in
    OpenMP emulation. It corresponds to the outermost loop.  */
#define P4A_vp_2 P4A_vp_coordinate[2]


/** @addtogroup P4A_cuda_kernel_call

    @{
*/

/** Call a CUDA like kernel with an OpenMP implementation.

    An API for full call control. For simpler usage: @see
    P4A_call_accel_kernel_1d, @see P4A_call_accel_kernel_2d, @see
    P4A_call_accel_kernel_3d

    This is not yet implemented. Ask for it if you really want it (that is
    you need the dim3 grids of blocks and blocks of threads). Right now
    only the functions useful for codegeneration from PIPS has been
    implemented.
*/
#define P4A_call_accel_kernel(context, parameters) \
  error "P4A_call_accel_kernel not yet implemented in OpenMP"
/* @} */


/** CUDA kernel invocation with an OpenMP implementation.

    Generate something like "kernel<<<block_dimension,thread_dimension>>>

    This is not yet implemented.
*/
#define P4A_call_accel_kernel_context(kernel, ...) \
  error "P4A_call_accel_kernel_context not yet implemented in OpenMP"


/** Add CUDA kernel parameters for invocation with an OpenMP
    implementation.

    This is not yet implemented.
*/
#define P4A_call_accel_kernel_parameters(...) \
  error "P4A_call_accel_kernel_parameters not yet implemented in OpenMP"


/** Creation of block and thread descriptors */

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


/** @addtogroup P4A_cuda_kernel_call

    @{
*/

/** Call a kernel in a 1-dimension parallel loop with OpenMP emulation.

    Translate it into an OpenMP parallel loop

    @param[in] kernel to call

    @param[in] P4A_n_iter_0 is the number of iterations

    @param ... the following parameters are given to the kernel
*/
#define P4A_call_accel_kernel_1d(kernel, P4A_n_iter_0, ...)		\
  P4A_skip_debug(3,P4A_dump_location();)					\
  P4A_skip_debug(1,P4A_dump_message("P4A_call_accel_kernel_1d(%d) of \"%s\"\n", P4A_n_iter_0, #kernel);)	\
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

    @param[in] P4A_n_iter_0 is the number of iterations in the first
    dimension, corresponding to deepest loop

    @param[in] P4A_n_iter_1 is the number of iterations in the second
    dimension, corresponding to outermost loop

    @param ... following parameters are given to the kernel
*/
#define P4A_call_accel_kernel_2d(kernel, P4A_n_iter_0, P4A_n_iter_1, ...) \
  P4A_skip_debug(3,P4A_dump_location();)					\
  P4A_skip_debug(1,P4A_dump_message("P4A_call_accel_kernel_2d(%d,%d) of \"%s\"\n", P4A_n_iter_0, P4A_n_iter_1, #kernel);)	\
  _Pragma("omp parallel for")						\
    for(int P4A_index_1 = 0; P4A_index_1 < P4A_n_iter_1; P4A_index_1++) { \
      for(int P4A_index_0 = 0; P4A_index_0 < P4A_n_iter_0; P4A_index_0++) { \
      P4A_vp_0 = P4A_index_0;						\
      P4A_vp_1 = P4A_index_1;						\
      P4A_vp_2 = 0;							\
      P4A_skip_debug(5,P4A_dump_message("%d %d\n", P4A_vp_0, P4A_vp_1));	\
      kernel(__VA_ARGS__);						\
    }									\
  }


/** Call a kernel in a 3-dimension parallel loop with OpenMP emulation.

    Translate it into an OpenMP parallel loop-nest

    @param[in] kernel to call

    @param[in] P4A_n_iter_0 is the number of iterations in the first
    dimension, corresponding to deepest loop

    @param[in] P4A_n_iter_1 is the number of iterations in the second
    dimension

    @param[in] P4A_n_iter_2 is the number of iterations in the third
    dimension, corresponding to outermost loop

    @param ... following parameters are given to the kernel
*/
#define P4A_call_accel_kernel_3d(kernel, P4A_n_iter_0, P4A_n_iter_1, P4A_n_iter_2, ...) \
  P4A_skip_debug(3,P4A_dump_location();)					\
  P4A_skip_debug(1,P4A_dump_message("P4A_call_accel_kernel_3d(%d,%d,%d) of \"%s\"\n", P4A_n_iter_0, P4A_n_iter_1, P4A_n_iter_2, #kernel);)	\
  _Pragma("omp parallel for")						\
  for(int P4A_index_2 = 0; P4A_index_2 < P4A_n_iter_2; P4A_index_2++) { \
    for(int P4A_index_1 = 0; P4A_index_1 < P4A_n_iter_1; P4A_index_1++) { \
      for(int P4A_index_0 = 0; P4A_index_0 < P4A_n_iter_0; P4A_index_0++) { \
        P4A_vp_0 = P4A_index_0;						\
        P4A_vp_1 = P4A_index_1;						\
        P4A_vp_2 = P4A_index_2;						\
        kernel(__VA_ARGS__);						\
      }									\
    }									\
  }

/** @} */

#endif
