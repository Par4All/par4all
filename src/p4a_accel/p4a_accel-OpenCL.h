/** @file

    API of Par4All C to OpenCL

    This file is an implementation of the OpenCL Par4All API.

    Funded by the FREIA (French ANR), TransMedi\@ (French Pôle de
    Compétitivité Images and Network) and SCALOPES (Artemis European
    Project project), with participation of MODENA (French Pôle de
    Compétitivité Mer Bretagne)

    "mailto:Stephanie.Even@enstb.org"
    "mailto:Ronan.Keryell@hpc-project.com"

    This work is done under MIT license.
*/

/* Do not include twice this file: */
#ifndef P4A_ACCEL_CL_H
#define P4A_ACCEL_CL_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <stdbool.h>
//#include <oclUtils.h>
#include <opencl.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <cl.h>

//#include <p4a_wrap-opencl.h>
#include "p4a_include-OpenCL.h"

#define P4A_log_and_exit(code,...)               \
  do {						 \
  fprintf(stdout,__VA_ARGS__);			 \
  exit(code);					 \
  } while(0)

#ifdef P4A_DEBUG
#define P4A_log(...)               fprintf(stdout,__VA_ARGS__)
#else
#define P4A_log(...)   
#endif

#define P4A_error_to_string(error)     (char *)p4a_error_to_string(error)
#define P4A_test_execution(error)      p4a_error(error, __FILE__, __LINE__)
#define P4A_test_execution_with_message(message)	\
  p4a_message(message, __FILE__, __LINE__)

/** @defgroup P4A_cl_kernel_call Accelerator kernel call

    @{
*/

/** Parameters used to choose work_item allocation per work_group in OpenCL.

    This allow to choose the size of the group of work_item in 1,2 and 3D.

    If there is not enough iterations to fill an expected work_group size in a
    dimension, the work_group size is indeed the number of iteration in this
    dimension to avoid spoling OpenCL work_item here.

    It should be OK for big iteration spaces, but quite suboptimal for
    small ones.

    They can be redefined at compilation time with definition options such
    as -DP4A_CL_ITEM_Y_PER_GROUP_IN_2D=128, or given in p4a with
    --cc_flags=-DP4A_CL_ITEM_Y_PER_GROUP_IN_2D=16 for example.

    If you get arrors such as "P4A OpenCL kernel execution failed : too many 
    resources requested for launch.", there is not enough registers to run
    all the requested items of your group. So try to reduce them with 
    -DP4A_CL_ITEM_MAX=384 or less. That may need some trimming.

    There are unfortunately some hardware limits on the group size
    that appear at the programming level, so we should add another level
    of tiling.
    
*/
#ifndef P4A_CL_ITEM_MAX
/** The maximum number of work_items in a work_group */
#define P4A_CL_ITEM_MAX 512
//#define P4A_CL_ITEM_MAX CL_DEVICE_MAX_WORK_GROUP_SIZE
#endif

// The localWorkGroupSize
#ifndef P4A_CL_ITEM_PER_GROUP_IN_1D
/** There is a maximum of P4A_CL_ITEM_MAX items per group. Use
    them. May cause some trouble if too many resources per item are
    used. Change it with an option at compile time if that causes
    trouble: */
#define P4A_CL_ITEM_PER_GROUP_IN_1D P4A_CL_ITEM_MAX
#endif

#ifndef P4A_CL_ITEM_X_PER_GROUP_IN_2D
/** The item layout size per group in 2D organization asymptotically. If
    there are enough iterations along X and the amount of iteration is not
    that big in Y, P4A_create_2d_thread_descriptors() allocates all
    threads in X that seems to often leads to better performance if the
    memory accesses are rather according to the X axis: */
#define P4A_CL_ITEM_X_PER_GROUP_IN_2D (P4A_CL_ITEM_MAX/16)
#endif
#ifndef P4A_CL_ITEM_Y_PER_GROUP_IN_2D
#define P4A_CL_ITEM_Y_PER_GROUP_IN_2D (P4A_CL_ITEM_MAX/P4A_CL_ITEM_X_PER_GROUP_IN_2D)
#endif

#ifndef P4A_CL_ITEM_X_PER_GROUP_IN_3D
/** The thread layout size per block in 3D organization

    Well, the 3D is not really dealt in CL even it is described by a
    dim3...
*/
#define P4A_CL_ITEM_X_PER_GROUP_IN_3D 8
#endif
#ifndef P4A_CL_ITEM_Y_PER_GROUP_IN_3D
#define P4A_CL_ITEM_Y_PER_GROUP_IN_3D 8
#endif
#ifndef P4A_CL_ITEM_Z_PER_GROUP_IN_3D
#define P4A_CL_ITEM_Z_PER_GROUP_IN_3D 8
#endif

/** @} */

/** @defgroup P4A_init_CL Initialization of P4A C to CL

    @{
*/

/** Associate the program to the accelerator in CL
    Could be placed in a inlined function ...

    Initialized the use of the hardware accelerator

    Just initialize events for time measure right now.
*/
#define P4A_init_accel	                                                \
  do {                                                                  \
    /* Get an OpenCL platform : a set of devices available */		\
    /* The device_id is the selected device from his type */		\
    /* CL_DEVICE_TYPE_GPU */						\
    /*  CL_DEVICE_TYPE_CPU */						\
    cl_platform_id p4a_platform_id = NULL;				\
     p4a_global_error=clGetPlatformIDs(1, &p4a_platform_id, NULL);	\
     P4A_test_execution_with_message("clGetPlatformIDs");				\
     /* Get the devices,could be a collection of device */		\
     cl_device_id p4a_device_id = NULL;                                 \
     p4a_global_error=clGetDeviceIDs(p4a_platform_id,			\
				     CL_DEVICE_TYPE_GPU,		\
				     1,					\
				     &p4a_device_id,			\
				     NULL);				\
     P4A_test_execution_with_message("clGetDeviceIDs");			\
     /* Create the context */						\
     p4a_context=clCreateContext(0,/*const cl_context_properties *properties*/ \
				 1,/*cl_uint num_devices*/		\
				 &p4a_device_id,			\
				 NULL,/*CL_CALLBACK *pfn_notify*/	\
				 NULL,					\
				 &p4a_global_error);			\
     P4A_test_execution_with_message("clCreateContext");				\
     /* ... could query many device, we retain only the first one ... */ \
     /* Create a file allocated to the first device ...   */		\
     p4a_queue=clCreateCommandQueue(p4a_context,p4a_device_id,          \
				    p4a_queue_properties,		\
				    &p4a_global_error);			\
     P4A_test_execution_with_message("clCreateCommandQueue");		\
  } while (0)


/** Release the hardware accelerator in CL
*/
#ifdef P4A_PROFILING
#define P4A_release_accel					\
  do {								\
    if (p4a_time_tag) printf("Copy time : %f\n",p4a_time_copy);	\
    p4a_clean(EXIT_SUCCESS);					\
  } while (0)
#else
#define P4A_release_accel    p4a_clean(EXIT_SUCCESS)
#endif
    

/** @} */


/** @defgroup P4A_cl_time_measure Time execution measurement

    @{
*/

/** Start a timer on the accelerator
    
    Nothing to do in OpenCL.
    Timer functions are linked to events.
    An event is called as an argument of a specific function 
    (kernel call, mem copy).
    The end and start time are available only after the call
    and are retrieved from the event.
 */
#define P4A_accel_timer_start 

/** @} */

/** A declaration attribute of a hardware-accelerated kernel in CL
    called from the GPU it-self
*/
#define P4A_accel_kernel inline void

/** A declaration attribute of a hardware-accelerated kernel called from
    the host in CL */
#define P4A_accel_kernel_wrapper __kernel void

/** The address space visible for all functions. 
    Allocation in the global memory pool.
 */
#define P4A_accel_global_address __global

/** The address space in the global memory pool but in read-only mode.
 */
#define P4A_accel_constant_address __constant

/** The address space visible by all work-items in a work group.
    This is the <<shared>> memory in the CUDA architecture.
    Can't be initialized :
    * __local float a = 1; is not allowed
    * __local float a;
              a = 1;       is allowed.
 */
#define P4A_accel_local_address __local


/** Get the coordinate of the virtual processor in X (first) dimension in
    CL
*/
#define P4A_vp_0 get_global_id(0)

/** Get the coordinate of the virtual processor in Y (second) dimension in
    CL
*/
#define P4A_vp_1 get_global_id(1)

/** Get the coordinate of the virtual processor in Z (second) dimension in
    CL
*/
#define P4A_vp_2 get_global_id(2)


/** @defgroup P4A_cl_kernel_call

    @{
*/

/** Call a CL kernel on the accelerator.

    An API for full call control. For simpler usage: 
    @see P4A_call_accel_kernel_1d, 
    @see P4A_call_accel_kernel_2d, 
    @see P4A_call_accel_kernel_3d

    This transforms :

    P4A_call_accel_kernel((clEnqueueNDRangeKernel),
                          (p4a_queue,p4a_kernel,work_dim,NULL,P4A_block_descriptor,P4A_grid_descriptor,0,NULL,&p4a_event_execution));

    into:

    do { clEnqueueNDRangeKernel(p4a_queue,p4a_kernel,work_dim,NULL,P4A_block_descriptor,P4A_grid_descriptor,0,NULL,&p4a_event_execution); } while (0);
*/

#define P4A_call_accel_kernel(context, parameters)			\
  do {									\
    P4A_skip_debug(P4A_dump_location());				\
    P4A_skip_debug(P4A_dump_message("Invoking %s with %s\n",	        \
				    #context,				\
				    #parameters));                      \
    P4A_call_accel_kernel_context context				\
    P4A_call_accel_kernel_parameters parameters;			\
    P4A_test_execution_with_message("P4A OpenCL kernel execution");	\
  } while (0)



/* @} */


/** CL kernel invocation.
*/
#define P4A_call_accel_kernel_context(kernel, ...)	\
  kernel

/** Add CL kernel parameters for invocation.

    Simply add them in parenthesis.  Well, this could be done without
    variadic arguments... Just for fun. :-)
*/
#define P4A_call_accel_kernel_parameters(...)    \
  (__VA_ARGS__)

/** Creation of block and thread descriptors for CL */

/** Allocate the descriptors for a linear set of thread with a
    simple strip-mining for CL
*/
#define P4A_create_1d_thread_descriptors(grid_descriptor_name,		\
					 block_descriptor_name,		\
					 size)				\
  cl_uint work_dim = 1;							\
  /* Define the number of thread per block: */				\
  size_t block_descriptor_name = {P4A_min((int) size,			\
					  (int) P4A_CL_ITEM_PER_GROUP_IN_1D)}; \
  /* Define the ceil-rounded number of needed blocks of threads: */	\
  size_t grid_descriptor_name = {(int)size};				\
  /*P4A_skip_debug(P4A_dump_grid_descriptor(grid_descriptor_name);)*/	\
  /*P4A_skip_debug(P4A_dump_block_descriptor(block_descriptor_name);)*/ \


/** Allocate the descriptors for a 2D set of thread with a simple
    strip-mining in each dimension for CL

    globalWorkSize = global number of work-items.
    localWorkSize = number of work-items in a work-group.
    The dimension is equal to work_dim.

    The total number of work-items in a work-group equals
    localWorkSize[0] * ... * localWorkSize[work_dim-1]
    This value must be less than CL_DEVICE_MAX_WORK_GROUP_SIZE.

    localWorkSize can be set as NULL, and OpenCL implementation will 
    determine how to break the globalWorkSize.
  

*/
#define P4A_create_2d_thread_descriptors(grid_descriptor_name,		\
					 block_descriptor_name,		\
					 n_x_iter, n_y_iter)		\
  int p4a_block_x, p4a_block_y;						\
  /* Define the number of thread per block: */				\
  if (n_y_iter > 10000) {						\
    /* If we have a lot of interations in Y, use asymptotical block	\
       sizes: */							\
    p4a_block_x = P4A_CL_ITEM_X_PER_GROUP_IN_2D;			\
    p4a_block_y = P4A_CL_ITEM_Y_PER_GROUP_IN_2D;			\
  }									\
  else {								\
    /* Allocate a maximum of threads alog X axis (the warp dimension) for \
       better average efficiency: */					\
    p4a_block_x = P4A_min((int) n_x_iter,				\
			  (int) P4A_CL_ITEM_MAX);			\
    p4a_block_y = P4A_min((int) n_y_iter,				\
			  P4A_CL_ITEM_MAX/p4a_block_x);			\
  }									\
  cl_uint work_dim = 2;							\
  /* The localWorkSize argument for clEnqueueNDRangeKernel */		\
  size_t block_descriptor_name[]={(size_t)p4a_block_x,(size_t)p4a_block_y}; \
  /* The globalWorkSize argument for clEnqueueNDRangeKernel */		\
  /* Define the ceil-rounded number of needed blocks of threads: */	\
  size_t grid_descriptor_name[]={(size_t) n_x_iter,(size_t) n_y_iter};	\
  /*P4A_log("grid size : %d %d\n",n_x_iter,n_y_iter);*/			\
  /*P4A_log("block size : %d %d\n",p4a_block_x,p4a_block_y);*/		\


/** Dump a CL dim3 descriptor with an introduction message */
#define P4A_dump_descriptor(message, descriptor_name)			\
  P4A_dump_message(message "\""  #descriptor_name "\" of size %dx%dx%d\n", \
		   descriptor_name.x,					\
		   descriptor_name.y,					\
		   descriptor_name.z)

/** Dump a CL dim3 block descriptor */
#define P4A_dump_block_descriptor(descriptor_name)			\
  P4A_dump_descriptor("Creating thread block descriptor ", descriptor_name)


/** Dump a CL dim3 grid descriptor */
#define P4A_dump_grid_descriptor(descriptor_name)		\
  P4A_dump_descriptor("Creating grid of block descriptor ",	\
		      descriptor_name)


/** @addtogroup P4A_cuda_kernel_call

    @{
*/

/** Call a kernel in a 1-dimension parallel loop in CL

    @param[in] kernel to call

    @param[in] P4A_n_iter_0 is the number of iterations

    @param ... the following parameters are given to the kernel
*/
#define P4A_call_accel_kernel_1d(kernel, P4A_n_iter_0, ...)		\
  do {									\
    p4a_load_kernel_arguments(kernel,__VA_ARGS__);			\
    P4A_skip_debug(P4A_dump_message("Calling 1D kernel \"" #kernel      \
				    "\" of size %d\n",P4A_n_iter_0));	\
    P4A_create_1d_thread_descriptors(P4A_grid_descriptor,		\
				     P4A_block_descriptor,		\
				     P4A_n_iter_0);			\
    P4A_call_accel_kernel((clEnqueueNDRangeKernel),			\
			  (p4a_queue,p4a_kernel,work_dim,NULL,          \
			   &P4A_block_descriptor,&P4A_grid_descriptor,	\
			   0,NULL,&p4a_event_execution));		\
  } while (0)


/** Call a kernel in a 2-dimension parallel loop in CL

    The creation of the program depends on the kernel that is known only 
    from here. Could not place this initialization in the P4A_accel_init
    macro.

    @param[in] kernel to call

    @param[in] P4A_n_iter_0 is the number of iterations in the first dimension

    @param[in] P4A_n_iter_1 is the number of iterations in the first dimension

    @param ... following parameters are given to the kernel
*/
#define P4A_call_accel_kernel_2d(kernel, P4A_n_iter_0, P4A_n_iter_1, ...) \
  do {	                                                                \
    p4a_load_kernel_arguments(kernel,__VA_ARGS__);			\
    P4A_skip_debug(P4A_dump_message("Calling 2D kernel \"" #kernel      \
                   "\" of size (%dx%d)\n",P4A_n_iter_0, P4A_n_iter_1)); \
    P4A_create_2d_thread_descriptors(P4A_grid_descriptor,		\
				     P4A_block_descriptor,		\
				     P4A_n_iter_0, P4A_n_iter_1);	\
    p4a_time_tag = false;						\
    P4A_accel_timer_stop_and_float_measure();				\
    P4A_call_accel_kernel((clEnqueueNDRangeKernel),			\
                          (p4a_queue,p4a_kernel,work_dim,NULL,          \
			   P4A_grid_descriptor,P4A_block_descriptor,	\
			   0,NULL,&p4a_event_execution));		\
    p4a_time_tag = true;						\
  } while (0)

/** @} */


#endif //P4A_ACCEL_CL_H
