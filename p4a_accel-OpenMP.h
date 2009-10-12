/** @file

    API of Par4All C to OpenMP

    This file is an implementation of the C to CUDA Par4All API.

    Funded by the FREIA (French ANR), TransMedi\@ (French Pôle de
    Compétitivité Images and Network) and SCALOPES (Artemis European
    Project project)

    "mailto:Stephanie.Even@enstb.org"
    "mailto:Ronan.Keryell@hpc-project.com"
*/

/* License BSD */

#ifndef P4A_ACCEL_OPENMP_H
#define P4A_ACCEL_OPENMP_H

#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

struct timeval p4a_time_begin, p4a_time_end;

/** This is a global variable used to simulate P4A virtual processor
    coordinates in OpenMP because we need to pass a local variable to a
    function without passing it in the arguments.

    Use thead local storage to have it local to each OpenMP thread.
 */
extern __thread int P4A_vp_coordinate[P4A_vp_dim_max];


/** @defgroup P4A_OpenMP_time_measure Time execution measurement

    @{
*/

/** Start a timer on the accelerator */
#define P4A_accel_timer_start gettimeofday(&p4a_time_begin, NULL)
/** @} */

/** @defgroup P4A_init Initialization of P4A C to OpenMP

    @{
*/

/** Associate the program to the accelerator

    Initialized the use of the hardware accelerator

    Nothing to do
*/
#define P4A_init_accel


/** Release the hardware accelerator as OpenMP

    Nothing to do
 */
#define P4A_release_accel

/** @} */


/** A declaration attribute of a hardware-accelerated kernel

    Nothing by default
*/
#define P4A_accel_kernel


/** A declaration attribute of a hardware-accelerated kernel called from
    the host

    Nothing by default
*/
#define P4A_accel_kernel_wrapper


/** Get the coordinate of the virtual processor in X (first) dimension */
#define P4A_vp_x P4A_vp_coordinate[0]


/** Get the coordinate of the virtual processor in Y (second) dimension */
#define P4A_vp_y P4A_vp_coordinate[1]


/** Get the coordinate of the virtual processor in Z (second) dimension */
#define P4A_vp_z P4A_vp_coordinate[2]


/** @defgroup P4A_memory_allocation_copy Memory allocation and copy

    @{
*/

/** Allocate memory on the hardware accelerator

    For OpenMP it is on the host too

    @param[out] address is the address of a variable that is updated by
    this macro to contains the address of the allocated memory block

    @param[in] size is the size to allocate in bytes
*/
#define P4A_accel_malloc(address, size)		\
  *(void **)address = malloc(size)


/** Free memory on the hardware accelerator

    It is on the host too

    @param[in] address is the address of a previously allocated memory zone on
    the hardware accelerator
*/
#define P4A_accel_free(address)			\
  free(address)


/** Copy memory from the host to the hardware accelerator

    Since it is an OpenMP implementation, use standard memory copy
    operations

    Do not change the place of the pointers in the API. The host address
    is always in the first position...

    @param[in] host_address is the address of a source zone in the host memory

    @param[out] accel_address is the address of a destination zone in the
    accelerator memory

    @param[in] size is the size in bytes of the memory zone to copy
*/
#define P4A_copy_to_accel(host_address, accel_address, size)		\
  /* We can use memcpy() since we are sure there is no overlap */	\
  memcpy(accel_address, host_address, size)


/** Copy memory from the hardware accelerator to the host.

    Since it is an OpenMP implementation, use standard memory copy
    operations

    Do not change the place of the pointers in the API. The host address
    is always in the first position...

    @param[out] host_address is the address of a destination zone in the
    host memory

    @param[in] accel_address is the address of a source zone in the
    accelerator memory

    @param[in] size is the size in bytes of the memory zone to copy
*/
#define P4A_copy_from_accel(host_address, accel_address, size)	\
  /* Easy implementation since memcpy is symetrical :-) */	\
  P4A_copy_to_accel(accel_address, host_address, size)


/* @} */

/** @addtogroup P4A_cuda_kernel_call

    @{
*/

/** Call a CUDA kernel on the accelerator.

    An API for full call control. For simpler usage: @see
    P4A_CALL_ACCEL_KERNEL_1D, @see P4A_CALL_ACCEL_KERNEL_2D, @see
    P4A_CALL_ACCEL_KERNEL_3D

    This transform for example:

    P4A_CALL_ACCEL_KERNEL((pips_accel_1, 1, pips_accel_dimBlock_1),
                          (*accel_imagein_re, *accel_imagein_im));

    into:

    do { pips_accel_1<<<1, pips_accel_dimBlock_1>>> (*accel_imagein_re, *accel_imagein_im); __cutilCheckMsg ("P4A CUDA kernel execution failed", "init.cu", 58); } while (0);
*/
#define P4A_call_accel_kernel(context, parameters)

/* @} */


/** CUDA kernel invocation.

    Generate something like "kernel<<<block_dimension,thread_dimension>>>"
*/
#define P4A_call_accel_kernel_context(kernel, ...)


/** Add CUDA kernel parameters for invocation.

    Simply add them in parenthesis.  Well, this could be done without
    variadic arguments... Just for fun. :-)
*/
#define P4A_call_accel_kernel_parameters(...)


/** Creation of block and thread descriptors */

/** Allocate the descriptors for a linear set of thread with a
    simple strip-mining
*/
#define P4A_create_1d_thread_descriptors(block_descriptor_name,		\
					 grid_descriptor_name,		\
					 size)


/** Allocate the descriptors for a 2D set of thread with a simple
    strip-mining in each dimension
*/
#define P4A_create_2d_thread_descriptors(block_descriptor_name,		\
					 grid_descriptor_name,		\
					 n_x_iter, n_y_iter)


/** Dump a CUDA dim3 descriptor with an introduction message */
#define P4A_dump_descriptor(message, descriptor_name)


/** Dump a CUDA dim3 block descriptor */
#define P4A_dump_block_descriptor(descriptor_name)


/** Dump a CUDA dim3 grid descriptor */
#define P4A_dump_grid_descriptor(descriptor_name)


/** @addtogroup P4A_cuda_kernel_call

    @{
*/

/** Call a kernel in a 1-dimension parallel loop

    Translate it into an OpenMP parallel loop

    @param[in] kernel to call

    @param[in] size is the number of iterations

    @param ... the following parameters are given to the kernel
*/
#define P4A_call_accel_kernel_1d(kernel, size, ...)		\
  _Pragma("omp parallel for")					\
  for(int P4A_index_x = 0; P4A_index_x < size; P4A_index_x++) {	\
    P4A_vp_x = P4A_index_x;					\
    P4A_vp_y = 0;						\
    P4A_vp_z = 0;						\
    kernel(__VA_ARGS__);					\
  }


/** Call a kernel in a 2-dimension parallel loop

    Translate it into an OpenMP parallel loop-nest

    @param[in] kernel to call

    @param[in] n_x_iter is the number of iterations in the first dimension

    @param[in] n_y_iter is the number of iterations in the first dimension

    @param ... following parameters are given to the kernel
*/
#define P4A_call_accel_kernel_2d(kernel, n_x_iter, n_y_iter, ...)	\
  _Pragma("omp parallel for")						\
  for(int P4A_index_x = 0; P4A_index_x < n_x_iter; P4A_index_x++) {	\
    for(int P4A_index_y = 0; P4A_index_y < n_y_iter; P4A_index_y++) {	\
      P4A_vp_x = P4A_index_x;						\
      P4A_vp_y = P4A_index_y;						\
      P4A_vp_z = 0;							\
     kernel(__VA_ARGS__);						\
    }									\
  }

/** @} */

#endif
