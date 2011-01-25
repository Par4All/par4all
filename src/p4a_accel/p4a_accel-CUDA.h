/** @file

    API of Par4All C to CUDA

    This file is an implementation of the C to CUDA Par4All API.

    Funded by the FREIA (French ANR), TransMedi\@ (French Pôle de
    Compétitivité Images and Network) and SCALOPES (Artemis European
    Project project), with the support of the MODENA project (French
    Pôle de Compétitivité Mer Bretagne)

    "mailto:Stephanie.Even@enstb.org"
    "mailto:Ronan.Keryell@hpc-project.com"

    This work is done under MIT license.
*/

/* Do not include twice this file: */
#ifndef P4A_ACCEL_CUDA_H
#define P4A_ACCEL_CUDA_H

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

/** Includes the kernel and variable qualifiers.

    Some are used when launching the kernel.
*/
#include "p4a_accel_wrapper-CUDA.h"

/** @addtogroup P4A_log Debug messages and errors tracking.
    
    @{
*/

/** Test the result of CUDA execution 
 */
#define P4A_test_execution(error)					\
  do {									\
    if(cudaSuccess != error) {						\
      fprintf(stderr,"%s The runtime error is %s\n",			\
	      AT,cudaGetErrorString(error));				\
      exit(EXIT_FAILURE);						\
    }									\
  } while (0)

/** Test the result of CUDA execution with an additional message
 */
#define P4A_test_execution_with_message(message)			\
  do {									\
    cudaError_t error = cudaGetLastError();				\
    if(cudaSuccess != error) {						\
      fprintf(stderr,"%s - %s : %s\n",					\
	      AT,message,cudaGetErrorString(error));		\
      exit(EXIT_FAILURE);						\
    }									\
    error = cudaThreadSynchronize();					\
    if(cudaSuccess != error){						\
      fprintf(stderr,"%s - Error after ThreadSynchronize %s : %s\n",	\
	      AT,message, cudaGetErrorString(error));		\
      exit(EXIT_FAILURE);						\
    }									\
  } while (0)

/** @} */


/** @addtogroup P4A_time_measure Time measurement

    In CUDA, the time is measured between two asynchronous events.

    In CUDA, the sequence is
    - cudaEvent_t start,stop;
    - cudaEventCreate(&sart), cudaEventCreate(&stop)
    - cudaEventRecord(start,0) called at P4A_accel_timer_start
    - ... (some code)
    - cudaEventSynchronize(stop,0),
      called at P4A_accel_timer_stop_and_float_measure()
    - cudaEventElapsedTime(&elapsedTime,start,stop), 
      P4A_accel_timer_stop_and_float_measure()
    - cudaEventDestroy(sart), cudaEventDestroy(stop)

    @{

*/

/** In CUDA, event to record the beginning of a timer.

*/
extern cudaEvent_t p4a_start_event;

/** In CUDA, event to record the end of a timer. 
*/
extern cudaEvent_t p4a_stop_event;


/** Start a timer on the accelerator in CUDA.

    Records the p4a_start_event.

    Two versions are proposed :
    -# One from Stéphanie Even preserving the user point view which is invited
    to place P4A_accel_timer_start and P4A_accel_timer_stop_and_float_measure()
    everywhere he wants;
    -# One from Mehdi Amini which privilegies the P4A internal point of
       view with pre-formated and very detailed calls from P4A.

   The two versions have been preserved with two differents conditional, 
   respectively, P4A_PROFILING or P4A_TIMING.
*/

/** Stéphanie Even version : user point of view */
#ifdef P4A_PROFILING
#define P4A_accel_timer_start					\
  P4A_test_execution(cudaEventRecord(p4a_start_event, 0))
#else
#define P4A_accel_timer_start /* Nothing */
#endif


/** Mehdi Amini version : P4A internal */

// Stop a timer on the accelerator in CUDA 
#define P4A_accel_timer_stop toolTestExec(cudaEventRecord(p4a_stop_event, 0))

#ifdef P4A_TIMING
// Set of routine for timing kernel executions
extern float p4a_timing_elapsedTime;

#define P4A_TIMING_accel_timer_start P4A_accel_timer_start

#define P4A_TIMING_accel_timer_stop		\
  { P4A_accel_timer_stop;			\
    cudaEventSynchronize(p4a_stop_event);	\
  }

#define P4A_TIMING_elapsed_time(elapsed)				\
  cudaEventElapsedTime(&elapsed, p4a_start_event, p4a_stop_event);

#define P4A_TIMING_get_elapsed_time		      \
  {   P4A_TIMING_elapsed_time(&p4a_timing_elapsedTime); \
    p4a_timing_elapsedTime; }

#define P4A_TIMING_display_elasped_time(msg)	   \
  { P4A_TIMING_elapsed_time(p4a_timing_elapsedTime);	\
    P4A_dump_message("Time for '%s' : %fms\n",		\
		     #msg,				\
		     p4a_timing_elapsedTime ); }

#else

#define P4A_TIMING_accel_timer_start
#define P4A_TIMING_accel_timer_stop
#define P4A_TIMING_elapsed_time(elapsed)
#define P4A_TIMING_get_elapsed_time
#define P4A_TIMING_display_elasped_time(msg)

#endif //P4A_TIMING

/** @} */

/** @addtogroup P4A_init_release Begin and end actions

    @{
*/

/** Associate the program to the accelerator in CUDA

    Initialized the use of the hardware accelerator

    Just initialize events for time measure right now.
*/
#ifdef P4A_PROFILING
#define P4A_init_accel						\
  do {								\
    P4A_test_execution(cudaEventCreate(&p4a_start_event));	\
    P4A_test_execution(cudaEventCreate(&p4a_stop_event));	\
  } while (0)
#else
/* Nothing*/
#define P4A_init_accel 
#endif

/** Release the hardware accelerator in CUDA

    Nothing to do
*/
#define P4A_release_accel

/** @} */

/** @addtogroup P4A_kernel Accelerator kernel call
    
    @{
*/


/** @addtogroup P4A_kernel_protoizer Kernel protoizer.
    
    @{
    
    In C for CUDA, a langage similar to the C, the prototype of the
    kernel is the classical signature in C; P4A_accel_kernel_wrapper
    is a kernel qualifier defined in p4a_accel_wrapper-CUDA.h.
*/

#define P4A_wrapper_proto(kernel, ...)		\
  P4A_accel_kernel_wrapper kernel(__VA_ARGS__)

/** @addtogroup P4A_kernel_context Kernel contextualization.

    The kernel is called over a grid of threads in CUDA.

    @{
    
*/

/** CUDA kernel invocation.

    Generate something like "kernel<<<block_dimension,thread_dimension>>>"
*/
#define P4A_call_accel_kernel_context(kernel, ...)	\
  kernel<<<__VA_ARGS__>>>

/** 
    @}
*/

/** @addtogroup P4A_kernel_parameters Parameters invocation

    @{

    In C for CUDA, this the call to the (__VA_ARGS__) list.
*/

/** 
    Add CUDA kernel parameters for invocation after the kernel context (kernel <<< >>>).

    Simply add them in parenthesis.  Well, this could be done without
    variadic arguments... Just for fun. :-)
*/
#define P4A_call_accel_kernel_parameters(...)	(__VA_ARGS__)

/** 
    @}
*/

/** 
    @addtogroup P4A_grid_descriptors Constructors of the the grid descriptor

    In CUDA, macros to generates the grid (number of group of thread)
    and threads (number of thread per multicore) in 1D, 2D and 3D as a
    function of global parameters adapted to the CUDA architecture
    (maximum of 512/1024 threads / multicore, one multicore is 8
    cores, number of multicore is total cores / 8).

    @{
    
*/

/** 
    Parameters used to choose thread allocation per blocks in CUDA
    
    This allow to choose the size of the blocks of threads in 1,2 and 3D.
    
    If there is not enough iterations to fill an expected block size in a
    dimension, the block size is indeed the number of iteration in this
    dimension to avoid spoling CUDA threads here.

    It should be OK for big iteration spaces, but quite suboptimal for
    small ones.

    They can be redefined at compilation time with definition options such
    as -DP4A_CUDA_THREAD_Y_PER_BLOCK_IN_2D=128, or given in p4a with
    --nvcc_flags=-DP4A_CUDA_THREAD_Y_PER_BLOCK_IN_2D=16 for example.

    If you get errors such as "P4A CUDA kernel execution failed : too many 
    resources requested for launch.", there is not enough registers to run
    all the requested threads of your block. So try to reduce them with 
    -DP4A_CUDA_THREAD_MAX=384 or less. That may need some trimming.

    There are unfortunately some hardware limits on the thread block size
    that appear at the programming level, so we should add another level
    of tiling.
*/

#ifndef P4A_CUDA_THREAD_MAX
/** 
    The maximum number of threads in a block of thread 
*/
#define P4A_CUDA_THREAD_MAX 512
#endif

#ifndef P4A_CUDA_THREAD_PER_BLOCK_IN_1D
/** 
    There is a maximum of P4A_CUDA_THREAD_MAX threads per block. Use
    them. May cause some trouble if too many resources per thread are
    used. Change it with an option at compile time if that causes
    trouble: 
*/
#define P4A_CUDA_THREAD_PER_BLOCK_IN_1D P4A_CUDA_THREAD_MAX
#endif

#ifndef P4A_CUDA_THREAD_X_PER_BLOCK_IN_2D
/** 
    The thread layout size per block in 2D organization asymptotically. If
    there are enough iterations along X and the amount of iteration is not
    that big in Y, P4A_create_2d_thread_descriptors() allocates all
    threads in X that seems to often leads to better performance if the
    memory accesses are rather according to the X axis: 
*/
#define P4A_CUDA_THREAD_X_PER_BLOCK_IN_2D (P4A_CUDA_THREAD_MAX/16)
#endif

#ifndef P4A_CUDA_THREAD_Y_PER_BLOCK_IN_2D
#define P4A_CUDA_THREAD_Y_PER_BLOCK_IN_2D (P4A_CUDA_THREAD_MAX/P4A_CUDA_THREAD_X_PER_BLOCK_IN_2D)
#endif

#ifndef P4A_CUDA_THREAD_X_PER_BLOCK_IN_3D
/** 
    The thread layout size per block in 3D organization

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

/** Creation of block and thread descriptors for CUDA */

/** 
    Allocate the descriptors for a linear set of thread with a
    simple strip-mining for CUDA
*/
#define P4A_create_1d_thread_descriptors(grid_descriptor_name,		\
					 block_descriptor_name,		\
					 size)				\
  /* Define the number of thread per block: */				\
  dim3 block_descriptor_name(P4A_min((int)size,				\
				     (int)P4A_CUDA_THREAD_PER_BLOCK_IN_1D)); \
  /* Define the ceil-rounded number of needed blocks of threads: */	\
  dim3 grid_descriptor_name((((int)size)+P4A_CUDA_THREAD_PER_BLOCK_IN_1D-1) \
			    /P4A_CUDA_THREAD_PER_BLOCK_IN_1D);		\
  P4A_skip_debug(P4A_dump_grid_descriptor(grid_descriptor_name);)	\
  P4A_skip_debug(P4A_dump_block_descriptor(block_descriptor_name);)


/** 
    Allocate the descriptors for a 2D set of thread with a simple
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


/** 
    Dump a CUDA dim3 descriptor with an introduction message 
*/
#define P4A_dump_descriptor(message, descriptor_name)			\
  P4A_dump_message(message "\""  #descriptor_name "\" of size %dx%dx%d\n", \
		   descriptor_name.x,					\
		   descriptor_name.y,					\
		   descriptor_name.z)

/** 
    Dump a CUDA dim3 block descriptor 
*/
#define P4A_dump_block_descriptor(descriptor_name)			\
  P4A_dump_descriptor("Creating thread block descriptor ", descriptor_name)


/** 
    Dump a CUDA dim3 grid descriptor 
*/
#define P4A_dump_grid_descriptor(descriptor_name)		\
  P4A_dump_descriptor("Creating grid of block descriptor ",	\
		      descriptor_name)

/** 
    @}
*/

/** @addtogroup P4A_kernel_call Kernel call constructor

    The kernel call itself.

    For simple usage
    @see P4A_call_accel_kernel_1d, 
    @see P4A_call_accel_kernel_2d, 
    @see P4A_call_accel_kernel_3d

    @{
*/

/** A CUDA kernel call on the accelerator. This transforms for example:

    P4A_call_accel_kernel((pips_accel_1, 1, pips_accel_dimBlock_1),
                          (*accel_imagein_re, *accel_imagein_im));

    into:

    do { 
       pips_accel_1<<<1, pips_accel_dimBlock_1>>> (*accel_imagein_re, 
                                                *accel_imagein_im); 
       P4A_test_execution_with_message ("P4A CUDA kernel execution failed", 
                                        "init.cu", 58); 
    } while (0)
 */

#define P4A_call_accel_kernel(context, parameters)			\
  do {									\
    P4A_skip_debug(P4A_dump_location());				\
    P4A_skip_debug(P4A_dump_message("Invoking kernel %s with %s\n",	\
				    #context,				\
				    #parameters));			\
    P4A_TIMING_accel_timer_start;					\
    P4A_call_accel_kernel_context context				\
    P4A_call_accel_kernel_parameters parameters;			\
    P4A_test_execution_with_message("P4A CUDA kernel execution failed"); \
    P4A_TIMING_accel_timer_stop;					\
    P4A_TIMING_display_elasped_time(context);				\
  } while (0)



/** Call a kernel in a 1-dimension parallel loop in CUDA

    @param[in] kernel to call

    @param[in] P4A_n_iter_0 is the number of iterations

    @param ... the following parameters are given to the kernel
*/
#define P4A_call_accel_kernel_1d(kernel, P4A_n_iter_0, ...)		\
  do {									\
    P4A_skip_debug(P4A_dump_message("Calling 1D kernel \"" #kernel	\
				    "\" of size %d\n",P4A_n_iter_0));	\
    P4A_create_1d_thread_descriptors(P4A_grid_descriptor,		\
				     P4A_block_descriptor,		\
				     P4A_n_iter_0);			\
    P4A_call_accel_kernel((kernel, P4A_grid_descriptor, P4A_block_descriptor), \
			  (__VA_ARGS__));				\
  } while (0)


/** 
    Call a kernel in a 2-dimension parallel loop in CUDA

    @param[in] kernel to call

    @param[in] P4A_n_iter_0 is the number of iterations in the first dimension

    @param[in] P4A_n_iter_1 is the number of iterations in the first dimension

    @param ... following parameters are given to the kernel
*/
#define P4A_call_accel_kernel_2d(kernel, P4A_n_iter_0, P4A_n_iter_1, ...) \
  do {									\
    P4A_skip_debug(P4A_dump_message("Calling 2D kernel \"" #kernel	\
				    "\" of size (%dx%d)\n",		\
				    P4A_n_iter_0, P4A_n_iter_1));	\
    P4A_create_2d_thread_descriptors(P4A_grid_descriptor,		\
				     P4A_block_descriptor,		\
				     P4A_n_iter_0, P4A_n_iter_1);	\
    P4A_call_accel_kernel((kernel, P4A_grid_descriptor, P4A_block_descriptor), \
			  (__VA_ARGS__));				\
  } while (0)

/** @} */
/** @} */


#endif //P4A_ACCEL_CUDA_H
