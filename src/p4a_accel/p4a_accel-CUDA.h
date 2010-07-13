/** @file

    API of Par4All C to CUDA

    This file is an implementation of the C to CUDA Par4All API.

    Funded by the FREIA (French ANR), TransMedi\@ (French Pôle de
    Compétitivité Images and Network) and SCALOPES (Artemis European
    Project project)

    "mailto:Stephanie.Even@enstb.org"
    "mailto:Ronan.Keryell@hpc-project.com"

    This work is done under MIT license.
*/

/* Do not include twice this file: */
#ifndef P4A_ACCEL_CUDA_H
#define P4A_ACCEL_CUDA_H

#include <cutil_inline.h>
#include <cuda.h>

/** @defgroup P4A_cuda_kernel_call Accelerator kernel call

    @{
*/

/** Parameters used to choose thread allocation per blocks in CUDA

    This allow to choose the size of the blocks of threads in 1,2 and 3D.

    If there is not enough iterations to fill an expected block size in a
    dimension, the block size is indeed the number of iteration in this
    dimension to avoid spoling CUDA threads here.

    It should be OK for big iteration spaces, but quite suboptimal for
    small ones.

    They can be redefined at compilation time with definition options such
    as -DP4A_CUDA_THREAD_Y_PER_BLOCK_IN_2D=128, or given in p4a with
    --nvcc_flags=-DP4A_CUDA_THREAD_Y_PER_BLOCK_IN_2D=16 for example.

    If you get arrors such as "cutilCheckMsg() CUTIL CUDA error : P4A CUDA
    kernel execution failed : too many resources requested for launch.",
    there is not enough registers to run all the requested threads of your
    block. So try to reduce them with -DP4A_CUDA_THREAD_MAX=384 or
    less. That may need some trimming.

    There are unfortunately some hardware limits on the thread block size
    that appear at the programming level, so we should add another level
    of tiling.
*/
#ifndef P4A_CUDA_THREAD_MAX
/** The maximum number of threads in a block of thread */
#define P4A_CUDA_THREAD_MAX 512
#endif

#ifndef P4A_CUDA_THREAD_PER_BLOCK_IN_1D
/** There is a maximum of P4A_CUDA_THREAD_MAX threads per block. Use
    them. May cause some trouble if too many resources per thread are
    used. Change it with an option at compile time if that causes
    trouble: */
#define P4A_CUDA_THREAD_PER_BLOCK_IN_1D P4A_CUDA_THREAD_MAX
//P4A_CUDA_THREAD_PER_BLOCK_IN_1D = 5,
#endif

#ifndef P4A_CUDA_THREAD_X_PER_BLOCK_IN_2D
/** The thread layout size per block in 2D organization asymptotically. If
    there are enough iterations along X and the amount of iteration is not
    that big in Y, P4A_create_2d_thread_descriptors() allocates all
    threads in X that seems to often leads to better performance if the
    memory accesses are rather according to the X axis: */
#define P4A_CUDA_THREAD_X_PER_BLOCK_IN_2D (P4A_CUDA_THREAD_MAX/16)
#endif
#ifndef P4A_CUDA_THREAD_Y_PER_BLOCK_IN_2D
#define P4A_CUDA_THREAD_Y_PER_BLOCK_IN_2D (P4A_CUDA_THREAD_MAX/P4A_CUDA_THREAD_X_PER_BLOCK_IN_2D)
#endif

#ifndef P4A_CUDA_THREAD_X_PER_BLOCK_IN_3D
/** The thread layout size per block in 3D organization

    Well, the 3D is not really dealt in CUDA even it is described by a
    dim3...
*/
#define P4A_CUDA_THREAD_X_PER_BLOCK_IN_3D 8
#endif
#ifndef P4A_CUDA_THREAD_Y_PER_BLOCK_IN_3D
#define P4A_CUDA_THREAD_Y_PER_BLOCK_IN_3D 8
#endif
#ifndef P4A_CUDA_THREAD_Z_PER_BLOCK_IN_3D
#define P4A_CUDA_THREAD_Z_PER_BLOCK_IN_3D 8
#endif

/** @} */

/** Events for timing in CUDA: */
extern cudaEvent_t p4a_start_event, p4a_stop_event;


/** @defgroup P4A_init_CUDA Initialization of P4A C to CUDA

    @{
*/

/** Associate the program to the accelerator in CUDA

    Initialized the use of the hardware accelerator

    Just initialize events for time measure right now.
*/
#define P4A_init_accel					\
  do {							\
    cutilSafeCall(cudaEventCreate(&p4a_start_event));	\
    cutilSafeCall(cudaEventCreate(&p4a_stop_event));	\
  } while (0);


/** Release the hardware accelerator in CUDA

    Nothing to do
*/
#define P4A_release_accel

/** @} */


/** @defgroup P4A_cuda_time_measure Time execution measurement

    @{
*/

/** Start a timer on the accelerator in CUDA */
#define P4A_accel_timer_start cutilSafeCall(cudaEventRecord(p4a_start_event, 0))

/** @} */



/** A declaration attribute of a hardware-accelerated kernel in CUDA
    called from the GPU it-self
*/
#define P4A_accel_kernel __device__

/** A declaration attribute of a hardware-accelerated kernel called from
    the host in CUDA */
#define P4A_accel_kernel_wrapper __global__


/** Get the coordinate of the virtual processor in X (first) dimension in
    CUDA
*/
#define P4A_vp_0 (blockIdx.x*blockDim.x + threadIdx.x)

/** Get the coordinate of the virtual processor in Y (second) dimension in
    CUDA
*/
#define P4A_vp_1 (blockIdx.y*blockDim.y + threadIdx.y)

/** Get the coordinate of the virtual processor in Z (second) dimension in
    CUDA
*/
#define P4A_vp_2 (blockIdx.z*blockDim.z + threadIdx.z)


/** @defgroup P4A_memory_allocation_copy Memory allocation and copy

    @{
*/

/** Allocate memory on the hardware accelerator in CUDA.

    @param[out] address is the address of a variable that is updated by
    this macro to contains the address of the allocated memory block

    @param[in] size is the size to allocate in bytes
 */
#define P4A_accel_malloc(address, size)			\
  cutilSafeCall(cudaMalloc((void **)address, size))


/** Free memory on the hardware accelerator in CUDA

    @param[in] address is the address of a previously allocated memory zone on
    the hardware accelerator
*/
#define P4A_accel_free(address)			\
  cutilSafeCall(cudaFree(address))


/** Copy memory from the host to the hardware accelerator in CUDA.

    Do not change the place of the pointers in the API. The host address
    is always in the first position...

    @param[in] host_address is the address of a source zone in the host memory

    @param[out] accel_address is the address of a destination zone in the
    accelerator memory

    @param[in] size is the size in bytes of the memory zone to copy
*/
#define P4A_copy_to_accel(host_address, accel_address, size)	\
  cutilSafeCall(cudaMemcpy(accel_address,			\
			   host_address,			\
			   size,				\
			   cudaMemcpyHostToDevice))

/** Copy memory from the hardware accelerator to the host in CUDA.

    Do not change the place of the pointers in the API. The host address
    is always in the first position...

    @param[out] host_address is the address of a destination zone in the
    host memory

    @param[in] accel_address is the address of a source zone in the
    accelerator memory

    @param[in] size is the size in bytes of the memory zone to copy
*/
#define P4A_copy_from_accel(host_address, accel_address, size)	\
  cutilSafeCall(cudaMemcpy(host_address,			\
			   accel_address,			\
			   size,				\
			   cudaMemcpyDeviceToHost))

/* @} */


/** @addtogroup P4A_cuda_kernel_call

    @{
*/

/** Call a CUDA kernel on the accelerator.

    An API for full call control. For simpler usage: @see
    P4A_call_accel_kernel_1d, @see P4A_call_accel_kernel_2d, @see
    P4A_call_accel_kernel_3d

    This transforms for example:

    P4A_call_accel_kernel((pips_accel_1, 1, pips_accel_dimBlock_1),
                          (*accel_imagein_re, *accel_imagein_im));

    into:

    do { pips_accel_1<<<1, pips_accel_dimBlock_1>>> (*accel_imagein_re, *accel_imagein_im); __cutilCheckMsg ("P4A CUDA kernel execution failed", "init.cu", 58); } while (0);
*/
#define P4A_call_accel_kernel(context, parameters)			\
  do {									\
    P4A_skip_debug(P4A_dump_location());				\
    P4A_skip_debug(P4A_dump_message("Invoking kernel %s with %s\n",	\
				    #context,				\
				    #parameters));			\
    P4A_call_accel_kernel_context context				\
    P4A_call_accel_kernel_parameters parameters;			\
    cutilCheckMsg("P4A CUDA kernel execution failed");			\
  } while (0)

/* @} */


/** CUDA kernel invocation.

    Generate something like "kernel<<<block_dimension,thread_dimension>>>"
*/
#define P4A_call_accel_kernel_context(kernel, ...)	\
  kernel<<<__VA_ARGS__>>>


/** Add CUDA kernel parameters for invocation.

    Simply add them in parenthesis.  Well, this could be done without
    variadic arguments... Just for fun. :-)
*/
#define P4A_call_accel_kernel_parameters(...)	\
  (__VA_ARGS__)


/** Creation of block and thread descriptors for CUDA */

/** Allocate the descriptors for a linear set of thread with a
    simple strip-mining for CUDA
*/
#define P4A_create_1d_thread_descriptors(grid_descriptor_name,		\
					 block_descriptor_name,		\
					 size)				\
  /* Define the number of thread per block: */				\
  dim3 block_descriptor_name(P4A_min((int) size,			\
				     (int) P4A_CUDA_THREAD_PER_BLOCK_IN_1D)); \
  /* Define the ceil-rounded number of needed blocks of threads: */	\
  dim3 grid_descriptor_name((((int)size) + P4A_CUDA_THREAD_PER_BLOCK_IN_1D - 1)/P4A_CUDA_THREAD_PER_BLOCK_IN_1D); \
  P4A_skip_debug(P4A_dump_grid_descriptor(grid_descriptor_name);)	\
  P4A_skip_debug(P4A_dump_block_descriptor(block_descriptor_name);)


/** Allocate the descriptors for a 2D set of thread with a simple
    strip-mining in each dimension for CUDA
*/
#define P4A_create_2d_thread_descriptors(grid_descriptor_name,		\
					 block_descriptor_name,		\
					 n_x_iter, n_y_iter)		\
  int p4a_block_x, p4a_block_y;						\
  /* Define the number of thread per block: */				\
									\
  if (n_y_iter > 10000) {						\
    /* If we have a lot of interations in Y, use asymptotical block	\
       sizes: */							\
    p4a_block_x = P4A_CUDA_THREAD_X_PER_BLOCK_IN_2D;			\
    p4a_block_y = P4A_CUDA_THREAD_Y_PER_BLOCK_IN_2D;			\
  }									\
  else {								\
    /* Allocate a maximum of threads alog X axis (the warp dimension) for \
       better average efficiency: */					\
    p4a_block_x = P4A_min((int) n_x_iter,				\
                          (int) P4A_CUDA_THREAD_MAX);			\
    p4a_block_y = P4A_min((int) n_y_iter,				\
                          P4A_CUDA_THREAD_MAX/p4a_block_x);		\
  }									\
  dim3 block_descriptor_name(p4a_block_x, p4a_block_y);			\
  /* Define the ceil-rounded number of needed blocks of threads: */	\
  dim3 grid_descriptor_name((((int) n_x_iter) + p4a_block_x - 1)/p4a_block_x, \
			    (((int) n_y_iter) + p4a_block_y - 1)/p4a_block_y); \
  P4A_skip_debug(P4A_dump_grid_descriptor(grid_descriptor_name);)	\
  P4A_skip_debug(P4A_dump_block_descriptor(block_descriptor_name);)


/** Dump a CUDA dim3 descriptor with an introduction message */
#define P4A_dump_descriptor(message, descriptor_name)			\
  P4A_dump_message(message "\""  #descriptor_name "\" of size %dx%dx%d\n", \
		   descriptor_name.x,					\
		   descriptor_name.y,					\
		   descriptor_name.z)

/** Dump a CUDA dim3 block descriptor */
#define P4A_dump_block_descriptor(descriptor_name)			\
  P4A_dump_descriptor("Creating thread block descriptor ", descriptor_name)


/** Dump a CUDA dim3 grid descriptor */
#define P4A_dump_grid_descriptor(descriptor_name)		\
  P4A_dump_descriptor("Creating grid of block descriptor ",	\
		      descriptor_name)


/** @addtogroup P4A_cuda_kernel_call

    @{
*/

/** Call a kernel in a 1-dimension parallel loop in CUDA

    @param[in] kernel to call

    @param[in] P4A_n_iter_0 is the number of iterations

    @param ... the following parameters are given to the kernel
*/
#define P4A_call_accel_kernel_1d(kernel, P4A_n_iter_0, ...)		\
  do {									\
    P4A_skip_debug(P4A_dump_message("Calling 1D kernel \"" #kernel "\" of size %d\n",	\
                   P4A_n_iter_0));					\
    P4A_create_1d_thread_descriptors(P4A_grid_descriptor,		\
				     P4A_block_descriptor,		\
				     P4A_n_iter_0);			\
    P4A_call_accel_kernel((kernel, P4A_grid_descriptor, P4A_block_descriptor), \
			  (__VA_ARGS__));				\
  } while (0)


/** Call a kernel in a 2-dimension parallel loop in CUDA

    @param[in] kernel to call

    @param[in] P4A_n_iter_0 is the number of iterations in the first dimension

    @param[in] P4A_n_iter_1 is the number of iterations in the first dimension

    @param ... following parameters are given to the kernel
*/
#define P4A_call_accel_kernel_2d(kernel, P4A_n_iter_0, P4A_n_iter_1, ...) \
  do {									\
    P4A_skip_debug(P4A_dump_message("Calling 2D kernel \"" #kernel "\" of size (%dx%d)\n", \
                   P4A_n_iter_0, P4A_n_iter_1));			\
    P4A_create_2d_thread_descriptors(P4A_grid_descriptor,		\
				     P4A_block_descriptor,		\
				     P4A_n_iter_0, P4A_n_iter_1);	\
    P4A_call_accel_kernel((kernel, P4A_grid_descriptor, P4A_block_descriptor), \
			  (__VA_ARGS__));				\
  } while (0)

/** @} */

#endif //P4A_ACCEL_CUDA_H
