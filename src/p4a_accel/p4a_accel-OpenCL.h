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
#ifndef P4A_ACCEL_OPENCL_H
#define P4A_ACCEL_OPENCL_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <stdbool.h>
#include <opencl.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/time.h>

// Includes the kernel and variable qualifiers.
// Some are used when launching the kernel.
#include "p4a_accel_wrapper-OpenCL.h"

/** @addtogroup P4A_log Debug messages and errors tracking.
    
    @{
*/

/** Global error in absence of a getLastError in OpenCL */
extern cl_int p4a_global_error;

/** Prototypes for the functions that converts OpenCL error codes to
string. This function does not exists by default in OpenCL standard.
 */
char * p4a_error_to_string(int error);

/** Prototype for a clean up function.

    A function is prefered to a MACRO definition as the clean up
    function is called to exit after error.
 */
void   p4a_clean(int exitCode);

/** @author : Based on the CUDA version from Grégoire Péan. 
              Picked and adapted to OpenCL by Stéphanie Even

    Error display based on OpenCL error codes.

    Each OpenCL function returns an error information.
 */
#define P4A_test_execution(error)					\
  do {									\
    if(CL_SUCCESS != error) {						\
      fprintf(stderr, "%s - The runtime error is %s\n",AT,		\
	      (char *)p4a_error_to_string(error));			\
      p4a_clean(EXIT_FAILURE);						\
    }									\
  } while (0)

/** @author : Based on the CUDA version from Grégoire Péan. 
              Picked and adapted to OpenCL by Stéphanie Even

    Error display with message based on OpenCL error codes.
 */
#define P4A_test_execution_with_message(message)			\
  do {									\
    if (CL_SUCCESS != p4a_global_error) {				\
      fprintf(stderr, "%s - Failed - %s : %s\n",AT,message,		\
	      p4a_error_to_string(p4a_global_error));			\
      p4a_clean(EXIT_FAILURE);						\
    }									\
  } while (0)

/** @} */


/** @addtogroup P4A_time_measure Time measurement

    In OpenCL, the time is measured for single call of OpenCL
    functions via a unique event.

    In OpenCL, only one event is defined, in opposition to CUDA
    where the time is measured between two events record. The is
    called as the last argument in OpenCL functions like
    clEnqueueNDRangeKernel for a kernel execution,
    clEnqueueWriteBuffer or clEnqueueReadBuffer for copy. The event
    records also the time elapsed during the execution of single
    specific function. The p4a_time global variable has been defined
    to cumulate the total time between the P4A_accel_timer_start call
    and the P4A_accel_timer_stop_and_float_measure() call. The
    p4a_time global variable is initialized at 0 at each call to the
    P4A_accel_timer_start.

    @{
*/

/** A tag is required to enable profiling in an OpenCL command queue
    - if 0 : no profiling enabled
    - if CL_QUEUE_PROFILING_ENABLE : profiling enabled.

    Must be defined at the creation of the command queue

    Because the context and the queue are created in the p4a_init,
    when p4a_accel_timer_start is called everywhere the
    p4a_queue_properties is defined as global variable. It is
    initialized in the p4a_accel.c and not here.
 */
extern cl_command_queue_properties p4a_queue_properties;

/** Cumulated time in OpenCL.  

    In OpenCL, time is measured after each call of a
    function (clEnqueueNDRangeKernel for a kernel execution,
    clEnqueueWriteBuffer or clEnqueueReadBuffer for copy).  

    We also need to cumulate each single time.
 */
extern double p4a_time;
extern double p4a_host_time;

/** Event for timing in OpenCL */
extern cl_event p4a_event;

/** A tag to identify if timer_stop is called from inside P4A or from outside */
extern bool timer_call_from_p4a;

/** A tag to identify if timer_stop is called from accel or host */
extern bool P4A_TIMING_fromHost;

/** The absolute time recorded when the timer is started

    To record times for OpenCL functions executed in the host (kernel load, ...)
 */
extern struct timeval p4a_time_begin; 

/** The absolute time recorded when the timer ends

    To record times for OpenCL functions executed in the host (kernel load, ...)
 */
extern struct timeval p4a_time_end;

/** Start a timer on the accelerator in OpenCL.
    
    Initialize the global time variable that will cumulates the timer.
    in the case of external user that is allowed to place the begin and
    end commands everywhere she/he wants.
 */

#ifdef P4A_PROFILING
#define P4A_accel_timer_start	\
  p4a_time = 0.;		\
  p4a_host_time = 0.		
  
#else
#define P4A_accel_timer_start 
#endif

// Stop a timer on the accelerator 
/**  
     No need in OpenCL since the timer is linked to instances of 
     OpenCL functions.
*/
#define P4A_accel_timer_stop			\
  gettimeofday(&p4a_time_end, NULL)

#ifdef P4A_TIMING        
// Set of routine for timing kernel executions
extern float p4a_timing_elapsedTime;

#define P4A_TIMING_accel_timer_start gettimeofday(&p4a_time_begin, NULL)

#define P4A_TIMING_accel_timer_stop P4A_accel_timer_stop

/** Elapsed time :
    -# from OpenCL functions launched in the accelerator
    -# Otherwise, function launched in the host (use of the gettimeofday)

    The tag P4A_TIMING_fromHost is switch off from here.
    At the moment it is switch on from p4a_load_kernel in p4a_accel.c.
 */
#define P4A_TIMING_elapsed_time(elapsed)				\
  {									\
    if (!P4A_TIMING_fromHost)    {					\
      cl_ulong start,end;						\
      clWaitForEvents(1, &p4a_event);					\
      P4A_test_execution(clGetEventProfilingInfo(p4a_event,		\
						 CL_PROFILING_COMMAND_END, \
						 sizeof(cl_ulong),	\
						 &end,NULL));		\
      P4A_test_execution(clGetEventProfilingInfo(p4a_event,		\
						 CL_PROFILING_COMMAND_START, \
						 sizeof(cl_ulong),	\
						 &start,NULL));		\
      /* Time in ms */							\
      elapsed = (float)(end - start)*1.0e-6;				\
    }									\
    else { 								\
      elapsed = (p4a_time_end.tv_sec - p4a_time_begin.tv_sec)		\
	+ (p4a_time_end.tv_usec - p4a_time_begin.tv_usec)*1e-6;		\
      elapsed *= 1e3;							\
      P4A_TIMING_fromHost = false;					\
    }									\
  }

#define P4A_TIMING_get_elapsed_time			\
  P4A_TIMING_elapsed_time(p4a_timing_elapsedTime)

#define P4A_TIMING_display_elasped_time(msg)		\
  {							\
    P4A_TIMING_elapsed_time(p4a_timing_elapsedTime);	\
    P4A_dump_message("Time for '%s' : %f ms\n",		\
		     #msg,				\
		     p4a_timing_elapsedTime );		\
  }

#else

#ifdef P4A_PROFILING
#define P4A_TIMING_accel_timer_start gettimeofday(&p4a_time_begin, NULL)

#else
#define P4A_TIMING_accel_timer_start
#endif

#define P4A_TIMING_accel_timer_stop
#define P4A_TIMING_elapsed_time(elapsed)
#define P4A_TIMING_get_elapsed_time
#define P4A_TIMING_display_elasped_time(msg)

#endif //P4A_TIMING



/** @} */

/** @addtogroup P4A_init_release Begin and end actions 

    @{
*/

/** The OpenCL context ~ a CPU process 
- a module
- data
- specific memory address 
*/
extern cl_context p4a_context;
/** The OpenCL command queue, that allows delayed execution
 */
extern cl_command_queue p4a_queue;

/** OpenCL initialization implies to identifiy the platform and select the 
    device. In our case, only GPU device and, at the moment,
    only the first one is selected.

    Context and queue linked to the selected device are created.
*/
#define P4A_init_accel	                                                \
  /* Get an OpenCL platform : a set of devices available */		\
  /* The device_id is the selected device from his type */		\
  /* CL_DEVICE_TYPE_GPU */						\
  /* CL_DEVICE_TYPE_CPU */						\
  cl_platform_id p4a_platform_id = NULL;				\
  p4a_global_error=clGetPlatformIDs(1, &p4a_platform_id, NULL);		\
  P4A_test_execution_with_message("clGetPlatformIDs");			\
  /* Get the devices,could be a collection of device */			\
  cl_device_id p4a_device_id = NULL;					\
  p4a_global_error=clGetDeviceIDs(p4a_platform_id,			\
				  CL_DEVICE_TYPE_GPU,			\
				  1,					\
				  &p4a_device_id,			\
				  NULL);				\
  P4A_test_execution_with_message("clGetDeviceIDs");			\
  /* Create the context */						\
  p4a_context=clCreateContext(0,/*const cl_context_properties *properties*/ \
			      1,/*cl_uint num_devices*/			\
			      &p4a_device_id,				\
			      NULL,/*CL_CALLBACK *pfn_notify*/		\
			      NULL,					\
			      &p4a_global_error);			\
  P4A_test_execution_with_message("clCreateContext");			\
  /* ... could query many device, we retain only the first one */	\
  /* Create a file allocated to the first device ...   */		\
  p4a_queue=clCreateCommandQueue(p4a_context,p4a_device_id,		\
				 p4a_queue_properties,			\
				 &p4a_global_error);			\
  P4A_test_execution_with_message("clCreateCommandQueue")		

/** Release the hardware accelerator in CL
 */
#define P4A_release_accel    p4a_clean(EXIT_SUCCESS)
    
/** @} */

/** @addtogroup P4A_kernel Accelerator kernel call

    @{
*/

/** @addtogroup P4A_kernel_protoizer Kernel protoizer

    @{
  
    In OpenCL, this is the name of the kernel declared as a string.
*/
#define P4A_wrapper_proto(kernel, ...)	 const char *kernel = #kernel

/** 
    @}
*/

/** @addtogroup P4A_kernel_context Kernel contextualization.

    The kernel is called over a work groups of work items in OpenCL.

    @{
*/

/** OpenCL invocation of clEnqueueNDRangeKernel.
*/
#define P4A_call_accel_kernel_context(kernel, ...)	kernel

/** 
    @}
*/

/** @addtogroup P4A_kernel_parameters Parameters invocation.

    @{

    Two types of call in OpenCL.

    ---

    Two types of call in OpenCL.

    -# OpenCL, the kernel is invoked via the clEnqueueNDRangeKernel.
    The call to parameters for clEnqueueNDRangeKernel is done in 
    P4A_call_accel_kernel_parameters(...).

    -# In a second step in OpenCL, the kernel parameters list must be enqueued
    via the clSetKernelArg function. Cmplex transformations are necessary
    to push the kernel arguments from a __VA_ARGS__ list.

    In OpenCL, the parameters are pushed one by one using the clSetKernelArg
    function taking the range, sizeof(type) and reference of the argument.
    The (__VA_ARGS__) should also be replaced by

    for (int i = 0;i < nargs;i++)

       clSetKernelArg(p4a_kernel,i,sizeof(arg),&arg);
    
    The P4A_argN(...) macro is called just before the P4A_call_accel_kernel
    invocation. Three steps are necessary:
    -# Counting the number of parameters;
    -# Construct the good P4A_arg##N(...) call by concatenation which 
       recursively calls the push of parameters one by one;
    -# Call the setKernelArg function with the good arguments.
*/

/** A global pointer to the current kernel selected
 */
extern cl_kernel p4a_kernel; 

/** The OpenCL parameters for invocation of clEnqueueNDRangeKernel
    (p4a_queue,p4a_kernel,work_dim,NULL,P4A_block_descriptor,P4A_grid_descriptor,0,NULL,&p4a_event).
    */
#define P4A_call_accel_kernel_parameters(...)    (__VA_ARGS__)

/** To solve the number of arguments from a __VA_ARGS__.

    See
    http://stackoverflow.com/questions/3868289/count-number-of-parameters-in-c-variable-argument-method-call.

    Invokes P4A_NARG_ with the parameters list followed by a reverse
    enumeration of number between 63 and 0.

    Let to see: the case of the 0 argument.

 */
#define P4A_NARG(...) P4A_NARG_(__VA_ARGS__,P4A_RSEQ_N())

/** Interprets the preceding list as a unique __VA_ARGS__
 */

#define P4A_NARG_(...) P4A_ARG_N(__VA_ARGS__)


/** Positional parameters are expands before inserting the macro
expansion.  The expansion of parametres list followed by the reverse
enumeration results in the returned value N that equals the exact
number of paramters.
 */
#define P4A_ARG_N( \
          _1, _2, _3, _4, _5, _6, _7, _8, _9,_10, \
         _11,_12,_13,_14,_15,_16,_17,_18,_19,_20, \
         _21,_22,_23,_24,_25,_26,_27,_28,_29,_30, \
         _31,_32,_33,_34,_35,_36,_37,_38,_39,_40, \
         _41,_42,_43,_44,_45,_46,_47,_48,_49,_50, \
         _51,_52,_53,_54,_55,_56,_57,_58,_59,_60, \
         _61,_62,_63,N,...) N

/** A reverse enumeration used to calculate the argument number up to 63.
 */
#define P4A_RSEQ_N() \
         63,62,61,60,                   \
         59,58,57,56,55,54,53,52,51,50, \
         49,48,47,46,45,44,43,42,41,40, \
         39,38,37,36,35,34,33,32,31,30, \
         29,28,27,26,25,24,23,22,21,20, \
         19,18,17,16,15,14,13,12,11,10, \
         9,8,7,6,5,4,3,2,1,0 

/** When the number of arguments is known, automatic call by
    concatenation to the function P4A_argn, where n is the number of
    arguments.
 */
#define P4A_argN(...) P4A_CONCATN(P4A_arg,P4A_NARG(__VA_ARGS__))(0,__VA_ARGS__)

/** Forcing the interpretation of two fields that have to be concatenate.
 */
#define P4A_CONCATN(a,b) P4A_CONCAT(a,b) 

/** Concatenation of two fields 
 */
#define P4A_CONCAT(a,b) a ## b

/** 21 parameters interpreter in OpenCl. The first parameter x is
interpreted via P4A_arg1(n,x,...); n is the rank of the parameter. The
20 last parameters are passed to P4A_arg20().*/
#define P4A_arg21(n,x,...)  P4A_arg1(n,x,...) P4A_arg20(n+1,__VA_ARGS__) 	
/** 20 parameters interpreter in OpenCl. The first parameter x is
interpreted via P4A_arg1(n,x,...); n is the rank of the parameter. The
19 last parameters are passed to P4A_arg19().*/
#define P4A_arg20(n,x,...)  P4A_arg1(n,x,...) P4A_arg19(n+1,__VA_ARGS__) 	
/** 19 parameters interpreter in OpenCl. The first parameter x is
interpreted via P4A_arg1(n,x,...); n is the rank of the parameter. The
18 last parameters are passed to P4A_arg18().*/
#define P4A_arg19(n,x,...)  P4A_arg1(n,x,...) P4A_arg18(n+1,__VA_ARGS__) 	
/** 18 parameters interpreter in OpenCl. The first parameter x is
interpreted via P4A_arg1(n,x,...); n is the rank of the parameter. The
17 last parameters are passed to P4A_arg17().*/
#define P4A_arg18(n,x,...)  P4A_arg1(n,x,...) P4A_arg17(n+1,__VA_ARGS__) 	
/** 17 parameters interpreter in OpenCl. The first parameter x is
interpreted via P4A_arg1(n,x,...); n is the rank of the parameter. The
16 last parameters are passed to P4A_arg16().*/
#define P4A_arg17(n,x,...)  P4A_arg1(n,x,...) P4A_arg16(n+1,__VA_ARGS__) 	
/** 16 parameters interpreter in OpenCl. The first parameter x is
interpreted via P4A_arg1(n,x,...); n is the rank of the parameter. The
15 last parameters are passed to P4A_arg15().*/
#define P4A_arg16(n,x,...)  P4A_arg1(n,x,...) P4A_arg15(n+1,__VA_ARGS__) 	
/** 15 parameters interpreter in OpenCl. The first parameter x is
interpreted via P4A_arg1(n,x,...); n is the rank of the parameter. The
14 last parameters are passed to P4A_arg14().*/
#define P4A_arg15(n,x,...)  P4A_arg1(n,x,...) P4A_arg14(n+1,__VA_ARGS__) 	
/** 14 parameters interpreter in OpenCl. The first parameter x is
interpreted via P4A_arg1(n,x,...); n is the rank of the parameter. The
13 last parameters are passed to P4A_arg13().*/
#define P4A_arg14(n,x,...)  P4A_arg1(n,x,...) P4A_arg13(n+1,__VA_ARGS__) 	
/** 13 parameters interpreter in OpenCl. The first parameter x is
interpreted via P4A_arg1(n,x,...); n is the rank of the parameter. The
13 last parameters are passed to P4A_arg12(). */
#define P4A_arg13(n,x,...)  P4A_arg1(n,x,...) P4A_arg12(n+1,__VA_ARGS__) 	
/** 12 parameters interpreter in OpenCl. The first parameter x is
interpreted via P4A_arg1(n,x,...); n is the rank of the parameter. The
13 last parameters are passed to P4A_arg11(). */
#define P4A_arg12(n,x,...)  P4A_arg1(n,x,...) P4A_arg11(n+1,__VA_ARGS__) 	
/** 11 parameters interpreter in OpenCl. The first parameter x is
interpreted via P4A_arg1(n,x,...); n is the rank of the parameter. The
13 last parameters are passed to P4A_arg10(). */
#define P4A_arg11(n,x,...)  P4A_arg1(n,x,...) P4A_arg10(n+1,__VA_ARGS__) 	
/** 10 parameters interpreter in OpenCl. The first parameter x is
interpreted via P4A_arg1(n,x,...); n is the rank of the parameter. The
13 last parameters are passed to P4A_arg9(). */
#define P4A_arg10(n,x,...)  P4A_arg1(n,x,...) P4A_arg9(n+1,__VA_ARGS__) 	
/** 9 parameters interpreter in OpenCl. The first parameter x is
interpreted via P4A_arg1(n,x,...); n is the rank of the parameter. The
13 last parameters are passed to P4A_arg8(). */
#define P4A_arg9(n,x,...)  P4A_arg1(n,x,...) P4A_arg8(n+1,__VA_ARGS__) 	
/** 8 parameters interpreter in OpenCl. The first parameter x is
interpreted via P4A_arg1(n,x,...); n is the rank of the parameter. The
13 last parameters are passed to P4A_arg7(). */
#define P4A_arg8(n,x,...)  P4A_arg1(n,x,...) P4A_arg7(n+1,__VA_ARGS__) 	
/** 7 parameters interpreter in OpenCl. The first parameter x is
interpreted via P4A_arg1(n,x,...); n is the rank of the parameter. The
13 last parameters are passed to P4A_arg6(). */
#define P4A_arg7(n,x,...)  P4A_arg1(n,x,...) P4A_arg6(n+1,__VA_ARGS__) 	
/** 6 parameters interpreter in OpenCl. The first parameter x is
interpreted via P4A_arg1(n,x,...); n is the rank of the parameter. The
13 last parameters are passed to P4A_arg5(). */
#define P4A_arg6(n,x,...)  P4A_arg1(n,x,...) P4A_arg5(n+1,__VA_ARGS__) 	
/** 5 parameters interpreter in OpenCl. The first parameter x is
interpreted via P4A_arg1(n,x,...); n is the rank of the parameter. The
13 last parameters are passed to P4A_arg4(). */
#define P4A_arg5(n,x,...)  P4A_arg1(n,x,...) P4A_arg4(n+1,__VA_ARGS__) 	
/** 4 parameters interpreter in OpenCl. The first parameter x is
interpreted via P4A_arg1(n,x,...); n is the rank of the parameter. The
13 last parameters are passed to P4A_arg3(). */
#define P4A_arg4(n,x,...)  P4A_arg1(n,x,...) P4A_arg3(n+1,__VA_ARGS__) 	
/** 3 parameters interpreter in OpenCl. The first parameter x is
interpreted via P4A_arg1(n,x,...); n is the rank of the parameter. The
13 last parameters are passed to P4A_arg2(). */
#define P4A_arg3(n,x,...)  P4A_arg1(n,x,...) P4A_arg2(n+1,__VA_ARGS__) 	
/** 2 parameters interpreter in OpenCl. The two parameters are
interpreted via P4A_arg1(n,x,...); n is the rank of the parameters. */
#define P4A_arg2(n,x,...)  P4A_arg1(n,x,...) P4A_arg1(n+1,__VA_ARGS__) 	

/** Final call of the clSetKernelArg function call with the good parameters.
    
    C++ version with template, to properly solve the type issue.
 */
#ifdef __cplusplus
template<typename ARG0> inline void p4a_setArguments(int i,char *s,ARG0 arg0) 
{
  //printf("%s : size %lu, rang %d\n",s,sizeof(arg0),i);
  p4a_global_error = clSetKernelArg(p4a_kernel,i,sizeof(arg0),&arg0);
  P4A_test_execution_with_message("clSetKernelArg");
}

/** Interpretation of one parameter in OpenCL. Call to the
clSetKernelArg via the C++ version of p4a_setArguments() such that the
parameters types are resolved.
 */
#define P4A_arg1(n,x,...)  p4a_setArguments(n,(char *)STRINGIFY(x),x);		
#else
/* See the C version of the p4a_setArguments in p4a_accel.c. The
    reference to the parameter is set as (void *) 
*/
void p4a_setArguments(int i,char *s,size_t size,void * ref_arg);

/** Interpretation of one parameter in OpenCL. Call to the
clSetKernelArg via the C version of p4a_setArguments() such that the
parameters types are resolved.
 */
#define P4A_arg1(n,x,...)  p4a_setArguments(n,STRINGIFY(x),sizeof(x),&x);  	
#endif 

/** 
    @}
*/

/** @addtogroup P4A_grid_descriptors Constructors of the the grid descriptor

    In OpenCL, the grid (total number of work items) and threads
    (local number of work item) descriptors are size_t [] of 1, 2 or 3
    dimension.

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

/** Creation of block and thread descriptors for OpenCL */

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
  P4A_skip_debug(P4A_dump_grid_descriptor(grid_descriptor_name);)	\
  P4A_skip_debug(P4A_dump_block_descriptor(block_descriptor_name);)


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
  if ((n_y_iter) > 10000) {						\
    /* If we have a lot of interations in Y, use asymptotical block	\
       sizes: */							\
    p4a_block_x = P4A_CL_ITEM_X_PER_GROUP_IN_2D;			\
    p4a_block_y = P4A_CL_ITEM_Y_PER_GROUP_IN_2D;			\
  }									\
  else {								\
    /* Allocate a maximum of threads along X axis (the warp dimension) for \
       better average efficiency: */					\
    p4a_block_x = P4A_min((int) (n_x_iter),				\
			  (int) P4A_CL_ITEM_MAX);			\
    p4a_block_y = P4A_min((int) (n_y_iter),				\
			  P4A_CL_ITEM_MAX/p4a_block_x);			\
  }									\
  cl_uint work_dim = 2;							\
  /* The localWorkSize argument for clEnqueueNDRangeKernel */		\
  size_t block_descriptor_name[]={(size_t)p4a_block_x,(size_t)p4a_block_y}; \
  /* The globalWorkSize argument for clEnqueueNDRangeKernel */		\
  /* Define the ceil-rounded number of needed blocks of threads: */	\
  /*size_t grid_descriptor_name[]={(size_t)(n_x_iter),(size_t)(n_y_iter)};*/ \
  int global_x = (n_x_iter+P4A_CL_ITEM_MAX-1)/P4A_CL_ITEM_MAX;		\
  global_x *= p4a_block_x;						\
  int global_y = n_y_iter;						\
  size_t grid_descriptor_name[]={(size_t)(global_x),(size_t)(global_y)}; \
  P4A_skip_debug(P4A_dump_grid_descriptor(grid_descriptor_name);)	\
  P4A_skip_debug(P4A_dump_block_descriptor(block_descriptor_name);)


/** Dump a CL dim3 descriptor with an introduction message */
#define P4A_dump_descriptor(message, descriptor_name)			\
  P4A_dump_message(message "\""  #descriptor_name "\" of size %dx%dx%d\n", \
		   descriptor_name.x,					\
		   descriptor_name.y,					\
		   descriptor_name.z)

/** Dump a CL dim3 block descriptor */
#define P4A_dump_block_descriptor(descriptor_name)			\
  P4A_dump_descriptor("Creating thread block descriptor ",#descriptor_name)


/** Dump a CL dim3 grid descriptor */
#define P4A_dump_grid_descriptor(descriptor_name)		\
  P4A_dump_descriptor("Creating grid of block descriptor ",#descriptor_name)


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

/** An OpenCL kernel is launched on the accelerator. This transforms :

    P4A_call_accel_kernel((clEnqueueNDRangeKernel),
                          (p4a_queue,p4a_kernel,work_dim,NULL,P4A_block_descriptor,P4A_grid_descriptor,0,NULL,&p4a_event));

    into:

    do { 
       clEnqueueNDRangeKernel(p4a_queue,p4a_kernel,work_dim,NULL,P4A_block_descriptor,P4A_grid_descriptor,0,NULL,&p4a_event); 
    } while (0);

    P4A_TIMING displays internal P4A timer at each call of the kernel.
    P4A_PROFILING cumulates the timer that is displayed only when the
    user calls the stop_timer function from outside P4A.
    Thus, the two calls inhibits each other but can't be mixed.

*/

#define P4A_call_accel_kernel(context, parameters)			\
  P4A_skip_debug(P4A_dump_location());					\
  P4A_skip_debug(P4A_dump_message("Invoking %s with %s\n",	        \
				  #context,				\
				  #parameters));			\
  P4A_TIMING_accel_timer_start;						\
  P4A_call_accel_kernel_context context					\
  P4A_call_accel_kernel_parameters parameters;				\
  P4A_test_execution_with_message("P4A OpenCL kernel execution");	\
  timer_call_from_p4a = true;						\
  P4A_accel_timer_stop_and_float_measure();				\
  timer_call_from_p4a = false;						\
  P4A_TIMING_accel_timer_stop;						\
  P4A_TIMING_display_elasped_time(context)				\

/** In OpenCL, each kernel is referenced by its name as a string, but
    is launched via a pointer. 
    The pointer is created at the first load via clCreateKernel.

    For many kernels, we have to memorise
    the correspondance between the name and the pointer i a structure.

    At each call of a kernel, we search it in a kernel list and
    initialise the global pointer p4a_kernel.

    The kernel structure.

    Possible to choose between the C++ version with a map or
    a classical list in C.
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
  //The constructor new_p4a_kernel is defined in the p4a_accel.c file
};

/** In C, pointer to the first element of the kernel list
 */
extern struct p4a_cl_kernel *p4a_kernels;
struct p4a_cl_kernel* new_p4a_kernel(const char *kernel);
struct p4a_cl_kernel *p4a_search_current_kernel(const char *kernel);
#endif

/** Prototype of the function that loads the kernel source as a string
    from an external file.
 */
void   p4a_load_kernel(const char *kernel,...);

/** Prototype of the function that creates the program and the kernel pointer.
 */
char * p4a_load_prog_source(char *cl_kernel_file,
			    const char *head,
			    size_t *length);

/** Call a kernel in a 1-dimension parallel loop in CL

    @param[in] kernel to call

    @param[in] P4A_n_iter_0 is the number of iterations

    @param ... the following parameters are given to the kernel
*/
#define P4A_call_accel_kernel_1d(kernel, P4A_n_iter_0, ...)		\
  do {									\
    P4A_TIMING_accel_timer_start;					\
    p4a_load_kernel(kernel,__VA_ARGS__);				\
    P4A_argN(__VA_ARGS__);						\
    timer_call_from_p4a = true;						\
    P4A_accel_timer_stop_and_float_measure();				\
    timer_call_from_p4a = false;					\
    P4A_TIMING_accel_timer_stop;					\
    P4A_TIMING_display_elasped_time(Kernel and arguments loading);	\
    P4A_skip_debug(P4A_dump_message("Calling 1D kernel \"" #kernel	\
				    "\" of size %d\n",P4A_n_iter_0));	\
    P4A_create_1d_thread_descriptors(P4A_grid_descriptor,		\
				     P4A_block_descriptor,		\
				     P4A_n_iter_0);			\
    P4A_call_accel_kernel((clEnqueueNDRangeKernel),			\
			  (p4a_queue,p4a_kernel,work_dim,NULL,		\
			   &P4A_block_descriptor,&P4A_grid_descriptor,	\
			   0,NULL,&p4a_event));				\
  } while(0)


/** Call a kernel in a 2-dimension parallel loop in CL

    The creation of the program depends on the kernel that is known only 
    from here. Could not place this initialization in the P4A_accel_init
    macro.

    do { } while(0) is needed to privatize local variable that can be
    duplicate if many kernels are launched at the same time.

    @param[in] kernel to call

    @param[in] P4A_n_iter_0 is the number of iterations in the first dimension

    @param[in] P4A_n_iter_1 is the number of iterations in the first dimension

    @param ... following parameters are given to the kernel
*/
#define P4A_call_accel_kernel_2d(kernel, P4A_n_iter_0, P4A_n_iter_1, ...) \
  do {									\
    P4A_TIMING_accel_timer_start;					\
    p4a_load_kernel(kernel,__VA_ARGS__);				\
    P4A_argN(__VA_ARGS__);						\
    P4A_TIMING_accel_timer_stop;					\
    P4A_TIMING_display_elasped_time(Kernel and arguments loading);	\
    timer_call_from_p4a = true;						\
    P4A_accel_timer_stop_and_float_measure();				\
    timer_call_from_p4a = false;					\
    P4A_skip_debug(P4A_dump_message("Calling 2D kernel \"" #kernel	\
				    "\" of size (%dx%d)\n",		\
				    P4A_n_iter_0, P4A_n_iter_1));	\
    P4A_create_2d_thread_descriptors(P4A_grid_descriptor,		\
				     P4A_block_descriptor,		\
				     P4A_n_iter_0, P4A_n_iter_1);	\
    P4A_call_accel_kernel((clEnqueueNDRangeKernel),			\
			  (p4a_queue,p4a_kernel,work_dim,NULL,		\
			   P4A_grid_descriptor,P4A_block_descriptor,	\
			   0,NULL,&p4a_event));				\
  } while (0)

/** 
    @}
*/

#endif //P4A_ACCEL_OPENCL_H
