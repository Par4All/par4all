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

#include <stdio.h>
#include <stdlib.h>
//#include <cutil_inline.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <p4a_stacksize_test.h>

#define toolTestExec(error)		checkErrorInline      	(error, __FILE__, __LINE__)
#define toolTestExecMessage(message)	checkErrorMessageInline	(message, __FILE__, __LINE__)

static inline void checkErrorInline(cudaError_t error, const char *currentFile, const int currentLine)
{
    if(cudaSuccess != error){
	fprintf(stderr, "File %s - Line %i - The runtime error is %s\n", currentFile, currentLine, cudaGetErrorString(error));
        exit(-1);
    }
}

static inline void checkErrorMessageInline(const char *errorMessage, const char *currentFile, const int currentLine)
{
    cudaError_t error = cudaGetLastError();
    if(cudaSuccess != error){
        fprintf(stderr, "File %s - Line %i - %s : %s\n", currentFile, currentLine, errorMessage, cudaGetErrorString(error));
        exit(-1);
    }

#ifdef P4A_DEBUG
    error = cudaThreadSynchronize();
    if(cudaSuccess != error){
	fprintf(stderr, "File %s - Line %i - Error after ThreadSynchronize %s : %s\n", currentFile, currentLine, errorMessage, cudaGetErrorString(error));
        exit(-1);
    }
#endif
}


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

    If you get arrors such as "P4A CUDA kernel execution failed : too many 
    resources requested for launch.", there is not enough registers to run
    all the requested threads of your block. So try to reduce them with 
    -DP4A_CUDA_THREAD_MAX=384 or less. That may need some trimming.

    There are unfortunately some hardware limits on the thread block size
    that appear at the programming level, so we should add another level
    of tiling.
*/
#ifndef P4A_CUDA_THREAD_MAX
/** The maximum number of threads in a block of thread */
#define P4A_CUDA_THREAD_MAX 512
#endif

#ifndef P4A_CUDA_MIN_BLOCKS
/** The minimum number of blocks */
#define P4A_CUDA_MIN_BLOCKS 32
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
    toolTestExec(cudaEventCreate(&p4a_start_event));	\
    toolTestExec(cudaEventCreate(&p4a_stop_event));	\
    checkStackSize(); \
  } while (0)


/** Release the hardware accelerator in CUDA

    Nothing to do
*/
#define P4A_release_accel

/** @} */


/** @defgroup P4A_cuda_time_measure Time execution measurement

    @{
*/

/** Start a timer on the accelerator in CUDA */
#define P4A_accel_timer_start toolTestExec(cudaEventRecord(p4a_start_event, 0))

/** Stop a timer on the accelerator in CUDA */
#define P4A_accel_timer_stop toolTestExec(cudaEventRecord(p4a_stop_event, 0))

#ifdef P4A_TIMING
// Set of routine for timing kernel executions
extern float p4a_timing_elapsedTime;
#define P4A_TIMING_accel_timer_start P4A_accel_timer_start
#define P4A_TIMING_accel_timer_stop \
{ P4A_accel_timer_stop; \
  cudaEventSynchronize(p4a_stop_event); \
}
#define P4A_TIMING_elapsed_time(elapsed) \
    cudaEventElapsedTime(&elapsed, p4a_start_event, p4a_stop_event);
#define P4A_TIMING_get_elapsed_time \
{   P4A_TIMING_elapsed_time(&p4a_timing_elapsedTime); \
    p4a_timing_elapsedTime; }
#define P4A_TIMING_display_elasped_time(msg) \
{ P4A_TIMING_elapsed_time(p4a_timing_elapsedTime); \
  P4A_dump_message("Time for '%s' : %fms\n",  \
                   #msg,      \
                   p4a_timing_elapsedTime ); }

#else
#define P4A_TIMING_accel_timer_start
#define P4A_TIMING_accel_timer_stop
#define P4A_TIMING_elapsed_time(elapsed)
#define P4A_TIMING_get_elapsed_time
#define P4A_TIMING_display_elasped_time(msg)
#endif //P4A_TIMING




/** @} */



/** A declaration attribute of a hardware-accelerated kernel in CUDA
    called from the GPU it-self
*/
#define P4A_accel_kernel __device__ void

/** A declaration attribute of a hardware-accelerated kernel called from
    the host in CUDA */
#define P4A_accel_kernel_wrapper __global__ void


/** Get the coordinate of the virtual processor in X (first) dimension in
    CUDA, corresponding to deepest loop to be memory friendly
*/
#define P4A_vp_0 (blockIdx.x*blockDim.x + threadIdx.x)

/** Get the coordinate of the virtual processor in Y (second) dimension in
    CUDA, corresponding to outermost loop
*/
#define P4A_vp_1 (blockIdx.y*blockDim.y + threadIdx.y)

/** Get the coordinate of the virtual processor in Z (second) dimension in
    CUDA
*/
#define P4A_vp_2 (blockIdx.z*blockDim.z + threadIdx.z)


/** @defgroup P4A_CUDA_memory_allocation_copy Memory allocation and copy for CUDA acceleration

    @{
*/

/** Copy a scalar from the hardware accelerator to the host

 It's a wrapper around CudaMemCopy*.

 Do not change the place of the pointers in the API. The host address
 is always first...

 @param[in] element_size is the size of one element of the array in
 byte

 @param[out] host_address point to the element on the host to write
 into

 @param[in] accel_address refer to the compact memory area to read
 data. In the general case, accel_address may be seen as a unique idea (FIFO)
 and not some address in some memory space.
 */
void P4A_copy_from_accel(size_t element_size,
    void *host_address,
    const void *accel_address);

/** Copy a scalar from the host to the hardware accelerator

 It's a wrapper around CudaMemCopy*.

 Do not change the place of the pointers in the API. The host address
 is always before the accel address...

 @param[in] element_size is the size of one element of the array in
 byte

 @param[out] host_address point to the element on the host to write
 into

 @param[in] accel_address refer to the compact memory area to read
 data. In the general case, accel_address may be seen as a unique idea (FIFO)
 and not some address in some memory space.
 */
void P4A_copy_to_accel(size_t element_size,
    void const*host_address,
    void *accel_address);

/** Function for copying memory from the hardware accelerator to a 1D array in
 the host.

 It's a wrapper around CudaMemCopy*.

 @param[in] element_size is the size of one element of the array in
 byte

 @param[in] d1_size is the number of elements in the array. It is not
 used but here for symmetry with functions of higher dimensionality

 @param[in] d1_block_size is the number of element to transfer

 @param[in] d1_offset is element order to start the transfer from (host side)

 @param[out] host_address point to the array on the host to write into

 @param[in] accel_address refer to the compact memory area to read
 data. In the general case, accel_address may be seen as a unique idea (FIFO)
 and not some address in some memory space.

 @return the host_address, by compatibility with memcpy().
 */
void P4A_copy_from_accel_1d(size_t element_size,
    size_t d1_size,
    size_t d1_block_size,
    size_t d1_offset,
    void *host_address,
    const void *accel_address);

/** Function for copying a 1D memory zone from the host to a compact memory
 zone in the hardware accelerator.

 This function could be quite simpler but is designed by symmetry with
 other functions.

 @param[in] element_size is the size of one element of the array in
 byte

 @param[in] d1_size is the number of elements in the array. It is not
 used but here for symmetry with functions of higher dimensionality

 @param[in] d1_block_size is the number of element to transfer

 @param[in] d1_offset is element order to start the transfer from

 @param[in] host_address point to the array on the host to read

 @param[out] accel_address refer to the compact memory area to write
 data. In the general case, accel_address may be seen as a unique idea
 (FIFO) and not some address in some memory space.
 */
void P4A_copy_to_accel_1d(size_t element_size,
    size_t d1_size,
    size_t d1_block_size,
    size_t d1_offset,
    const void *host_address,
    void *accel_address);

/** Function for copying memory from the hardware accelerator to a 2D array in
 the host.
 */
void P4A_copy_from_accel_2d(size_t element_size,
    size_t d1_size, size_t d2_size,
    size_t d1_block_size, size_t d2_block_size,
    size_t d1_offset, size_t d2_offset,
    void *host_address,
    const void *accel_address);

/** Function for copying a 2D memory zone from the host to a compact memory
 zone in the hardware accelerator.
 */
void P4A_copy_to_accel_2d(size_t element_size,
    size_t d1_size, size_t d2_size,
    size_t d1_block_size, size_t d2_block_size,
    size_t d1_offset, size_t d2_offset,
    const void *host_address,
    void *accel_address);

/** Function for copying memory from the hardware accelerator to a 3D array in
    the host.
*/
void P4A_copy_from_accel_3d(size_t element_size,
          size_t d1_size, size_t d2_size, size_t d3_size,
          size_t d1_block_size, size_t d2_block_size, size_t d3_block_size,
          size_t d1_offset, size_t d2_offset, size_t d3_offset,
          void *host_address,
          const void *accel_address);

/** Function for copying a 3D memory zone from the host to a compact memory
    zone in the hardware accelerator.
*/
void P4A_copy_to_accel_3d(size_t element_size,
        size_t d1_size, size_t d2_size, size_t d3_size,
        size_t d1_block_size, size_t d2_block_size, size_t d3_block_size,
        size_t d1_offset,   size_t d2_offset, size_t d3_offset,
        const void *host_address,
        void *accel_address);

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

    do { pips_accel_1<<<1, pips_accel_dimBlock_1>>> (*accel_imagein_re, *accel_imagein_im); toolTestExecMessage ("P4A CUDA kernel execution failed", "init.cu", 58); } while (0);
*/
#define P4A_call_accel_kernel(context, parameters)			\
  do {									\
    P4A_skip_debug(P4A_dump_location());				\
    P4A_skip_debug(P4A_dump_message("Invoking kernel %s with %s\n",	\
				    #context,				\
				    #parameters));			\
		P4A_TIMING_accel_timer_start; \
    P4A_call_accel_kernel_context context				\
    P4A_call_accel_kernel_parameters parameters;			\
    toolTestExecMessage("P4A CUDA kernel execution failed");			\
    P4A_TIMING_accel_timer_stop; \
    P4A_TIMING_display_elasped_time(context); \
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
  int tpb = P4A_min(P4A_CUDA_THREAD_PER_BLOCK_IN_1D,size/P4A_CUDA_MIN_BLOCKS); \
  tpb = tpb & ~31; /* Truncate so that we have a 32 multiple */ \
  tpb = P4A_max(tpb,32); \
  dim3 block_descriptor_name(P4A_min((int) size,			\
				     (int)tpb )); \
  /* Define the ceil-rounded number of needed blocks of threads: */	\
  dim3 grid_descriptor_name((((int)size) + tpb - 1)/tpb); \
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
    int tpb = P4A_min(P4A_CUDA_THREAD_MAX,(n_x_iter)*(n_y_iter)/P4A_CUDA_MIN_BLOCKS); \
    tpb = P4A_max(tpb,32); \
    p4a_block_x = P4A_min((int) n_x_iter,				\
                          (int) tpb);			\
    p4a_block_y = P4A_min((int) n_y_iter,				\
                          tpb/p4a_block_x);		\
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
    if(P4A_n_iter_0>0) { \
      P4A_skip_debug(P4A_dump_message("Calling 1D kernel \"" #kernel "\" of size %d\n",	\
                                      P4A_n_iter_0));					\
      P4A_create_1d_thread_descriptors(P4A_grid_descriptor,		\
                                       P4A_block_descriptor,		\
                                       P4A_n_iter_0);			\
      P4A_call_accel_kernel((kernel, P4A_grid_descriptor, P4A_block_descriptor), \
                            (__VA_ARGS__));				\
	  } \
  } while (0)


/** Call a kernel in a 2-dimension parallel loop in CUDA

    @param[in] kernel to call

    @param[in] P4A_n_iter_0 is the number of iterations in the first dimension

    @param[in] P4A_n_iter_1 is the number of iterations in the first dimension

    @param ... following parameters are given to the kernel
*/
#define P4A_call_accel_kernel_2d(kernel, P4A_n_iter_0, P4A_n_iter_1, ...) \
  do {									\
    if(P4A_n_iter_0>0 && P4A_n_iter_1) { \
      P4A_skip_debug(P4A_dump_message("Calling 2D kernel \"" #kernel "\" of size (%dx%d)\n", \
                                      P4A_n_iter_0, P4A_n_iter_1));			\
      P4A_create_2d_thread_descriptors(P4A_grid_descriptor,		\
                                       P4A_block_descriptor,		\
                                       P4A_n_iter_0, P4A_n_iter_1);	\
      P4A_call_accel_kernel((kernel, P4A_grid_descriptor, P4A_block_descriptor), \
			  (__VA_ARGS__));				\
    } \
  } while (0)

/** @} */




/** @defgroup Atomic operation macros definitions

    @{
*/
#define atomicAddInt(operand,val) atomicAdd((unsigned int *)operand,val)

// FIXME still others to define...

/** @} */


#endif //P4A_ACCEL_CUDA_H

