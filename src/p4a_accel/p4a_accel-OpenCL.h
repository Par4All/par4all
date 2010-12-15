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

/** A kernel list.
*/
#ifdef __cplusplus
#include <map>
#include <string>
struct p4a_cl_kernel {
  cl_kernel kernel;
  char *name;
  char *file_name;

  p4a_cl_kernel(const char *k) {
    name = (char *)strdup(k);
    kernel = NULL;  
    char* kernelFile;
    asprintf(&kernelFile,"./%s.cl",k);
    file_name = (char *)strdup(kernelFile);
  }
  ~p4a_cl_kernel() {}
};

extern std::map<std::string, struct p4a_cl_kernel * > p4a_kernels_map ;
#else
struct p4a_cl_kernel {
  cl_kernel kernel;
  char *name;
  char *file_name;
  struct p4a_cl_kernel *next;
  //The constructor new_ is defined in the p4a_accel.h file
};

/** Begin of the kernels list
 */
extern struct p4a_cl_kernel *p4a_kernels;

struct p4a_cl_kernel* new_p4a_kernel(const char *kernel);
struct p4a_cl_kernel *p4a_search_current_kernel(const char *kernel);
void p4a_setArguments(int i,char *s,size_t size,void * ref_arg);
#endif

/** A timer Tag to know when to print p4a_time_copy
 */
extern bool p4a_time_tag;
/** Total execution time.
    In OpenCL, time is mesured for each lunch of the kernel.
    We also need to cumulate all the particulate times.
 */
extern double p4a_time_execution;
/** Total time for copy of data (transfers host <-> device).
 */
extern double p4a_time_copy;
/** Global error in absence of a getLastError equivalent in OpenCL */
extern cl_int p4a_global_error;
/** Events for timing in CL: */
extern cl_event p4a_event_execution, p4a_event_copy;
/** The OpenCL context ~ a CPU process 
- a module
- data
- specific memory address 
*/
extern cl_context p4a_context;
/** The OpenCL command queue, that allows delayed execution
 */
extern cl_command_queue p4a_queue;

extern cl_command_queue_properties p4a_queue_properties;


/** The OpenCL programm composed of a set of modules
 */
extern cl_program p4a_program;  
/** The current module selected in the program
 */
extern cl_kernel p4a_kernel; 

char * p4a_error_to_string(int error);
void   p4a_clean(int exitCode);
void   p4a_error(cl_int error,const char *currentFile,const int currentLine);
void   p4a_message(const char *message,const char *currentFile,const int currentLine);
void   p4a_load_kernel(const char *kernel,...);
char * p4a_load_prog_source(char *cl_kernel_file,const char *head,size_t *length);

/** @}
 */

/** @defgroup P4A print error and log functions.

   @{
 */

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

/** @}
 */

/** @defgroup P4A macros to automatically treat __VA_ARGS__ calls

    @{
*/

/** To solve the number of arguments
    from 
    http://stackoverflow.com/questions/3868289/count-number-of-parameters-in-c-variable-argument-method-call
 */
#define P4A_NARG(...) \
         P4A_NARG_(__VA_ARGS__,P4A_RSEQ_N())

#define P4A_NARG_(...) \
         P4A_ARG_N(__VA_ARGS__)

#define P4A_ARG_N( \
          _1, _2, _3, _4, _5, _6, _7, _8, _9,_10, \
         _11,_12,_13,_14,_15,_16,_17,_18,_19,_20, \
         _21,_22,_23,_24,_25,_26,_27,_28,_29,_30, \
         _31,_32,_33,_34,_35,_36,_37,_38,_39,_40, \
         _41,_42,_43,_44,_45,_46,_47,_48,_49,_50, \
         _51,_52,_53,_54,_55,_56,_57,_58,_59,_60, \
         _61,_62,_63,N,...) N

#define P4A_RSEQ_N() \
         63,62,61,60,                   \
         59,58,57,56,55,54,53,52,51,50, \
         49,48,47,46,45,44,43,42,41,40, \
         39,38,37,36,35,34,33,32,31,30, \
         29,28,27,26,25,24,23,22,21,20, \
         19,18,17,16,15,14,13,12,11,10, \
         9,8,7,6,5,4,3,2,1,0 



#define P4A_CONCATN(a,b) P4A_CONCAT(a,b) 

#define P4A_CONCAT(a,b) a ## b

#define P4A_STRINGIFY(val) (char *)#val

#ifdef __cplusplus
template<typename ARG0> inline void p4a_setArguments(int i,char *s,ARG0 arg0) {
  printf("%s : size %lu, rang %d\n",s,sizeof(arg0),i);
  p4a_global_error = clSetKernelArg(p4a_kernel,i,sizeof(arg0),&arg0);
  P4A_test_execution_with_message("clSetKernelArg");
}
#define P4A_arg1(n,x,...)  p4a_setArguments(n,P4A_STRINGIFY(x),x);		
#else
/* See the function setArguments in p4a_accel.h */
#define P4A_arg1(n,x,...)  p4a_setArguments(n,P4A_STRINGIFY(x),sizeof(x),&x);		
#endif 


//#define P4A_arg1(n,x,...)  setArguments(n,STRINGIFY(x),x);		
#define P4A_arg2(n,x,...)  P4A_arg1(n,x,...) P4A_arg1(n+1,__VA_ARGS__) 	
#define P4A_arg3(n,x,...)  P4A_arg1(n,x,...) P4A_arg2(n+1,__VA_ARGS__) 	
#define P4A_arg4(n,x,...)  P4A_arg1(n,x,...) P4A_arg3(n+1,__VA_ARGS__) 	
#define P4A_arg5(n,x,...)  P4A_arg1(n,x,...) P4A_arg4(n+1,__VA_ARGS__) 	
#define P4A_arg6(n,x,...)  P4A_arg1(n,x,...) P4A_arg5(n+1,__VA_ARGS__) 	
#define P4A_arg7(n,x,...)  P4A_arg1(n,x,...) P4A_arg6(n+1,__VA_ARGS__) 	
#define P4A_arg8(n,x,...)  P4A_arg1(n,x,...) P4A_arg7(n+1,__VA_ARGS__) 	
#define P4A_arg9(n,x,...)  P4A_arg1(n,x,...) P4A_arg8(n+1,__VA_ARGS__) 	
#define P4A_arg10(n,x,...)  P4A_arg1(n,x,...) P4A_arg9(n+1,__VA_ARGS__) 	
#define P4A_arg11(n,x,...)  P4A_arg1(n,x,...) P4A_arg10(n+1,__VA_ARGS__) 	
#define P4A_arg12(n,x,...)  P4A_arg1(n,x,...) P4A_arg11(n+1,__VA_ARGS__) 	
#define P4A_arg13(n,x,...)  P4A_arg1(n,x,...) P4A_arg12(n+1,__VA_ARGS__) 	
#define P4A_arg14(n,x,...)  P4A_arg1(n,x,...) P4A_arg13(n+1,__VA_ARGS__) 	

#define P4A_argN(...)  P4A_CONCATN(P4A_arg,P4A_NARG(__VA_ARGS__))(0,__VA_ARGS__)


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
    p4a_load_kernel(kernel,__VA_ARGS__);				\
    P4A_argN(__VA_ARGS__);						\
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
    p4a_load_kernel(kernel,__VA_ARGS__);				\
    P4A_argN(__VA_ARGS__);						\
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
