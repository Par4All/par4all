/** @file

    API of Par4All C to OpenCL

    This file is an implementation of the OpenCL Par4All API.

    Funded by the FREIA (French ANR), TransMedi\@ (French Pôle de
    Compétitivité Images and Network) and SCALOPES (Artemis European
    Project project)

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
#include <oclUtils.h>
#include <opencl.h>
//#include <cl.h>


/** Global error in absence of a getLastError equivalent in OpenCL */
extern cl_int p4a_global_error;
/** Events for timing in CL: */
extern cl_event p4a_event;
/** The OpenCL context ~ a CPU process 
- a module
- data
- specific memory address 
*/
extern cl_context p4a_context;
/** The OpenCL command queue, that allows delayed execution
 */
extern cl_command_queue p4a_queue;
/** The selected device */
extern cl_device_id p4a_device_id;  
/** The OpenCL platform : a set of devices 
    The device_id is the selected device from his type
    CL_DEVICE_TYPE_GPU
    CL_DEVICE_TYPE_CPU
*/
extern cl_platform_id p4a_platform_id;  
/** The OpenCL programm composed of a set of modules
 */
extern cl_program p4a_program;  
/** The module selected in the program
 */
extern cl_kernel p4a_kernel;  
/** The sources of the modules composing the program
 */
extern const char* p4a_kernel_source;  

extern int p4a_args_count;
extern int *p4a_args_size;

#ifdef P4A_DEBUG
#define PRINT_LOG(...)               fprintf(stderr,__VA_ARGS__)
#else
#define PRINT_LOG(...)   
#endif


#define toolTestExec(error)          checkErrorInline(error, __FILE__, __LINE__)
#define toolTestExecMessage(message) checkErrorMessageInline(message, __FILE__, __LINE__)

inline void checkErrorInline(cl_int error, 
			     const char *currentFile, 
			     const int currentLine)
{
#ifndef P4A_DEBUG
    if(CL_SUCCESS != error) {
      //fprintf(stderr, "File %s - Line %i - The runtime error is %s\n", currentFile, currentLine, oclErrorString(error));
      fprintf(stderr, "File %s - Line %i - The runtime error is %d\n",
	      currentFile,currentLine,error);
      exit(-1);
    }
#else
    if(CL_SUCCESS != error) {
      //fprintf(stderr, "File %s - Line %i - The runtime error is %s\n", currentFile, currentLine, oclErrorString(error));
      fprintf(stderr, "File %s - Line %i - The runtime error is %d\n",
	      currentFile,currentLine,error);
      exit(-1);
    }
    else
      fprintf(stdout, "File %s - Line %i - The runtime is successful\n",
	      currentFile,currentLine);
#endif
}

// S. Even : I didn't find an equivalent for cudaGetLastError() at once
// To see later
inline void checkErrorMessageInline(const char *errorMessage, const char *currentFile, const int currentLine)
{
#ifndef P4A_DEBUG
  //cudaError_t error = cudaGetLastError();
  if(CL_SUCCESS != p4a_global_error){
    //fprintf(stderr, "File %s - Line %i - %s : %s\n", currentFile, currentLine, errorMessage, oclErrorString(p4a_global_error));
    fprintf(stderr, "File %s - Line %i - Failed - %s : %d\n", currentFile, currentLine, errorMessage, p4a_global_error);
    exit(-1);
  }
#else
  //cl_int error = cudaThreadSynchronize();
  clWaitForEvents(1, &p4a_event);
  if(CL_SUCCESS != p4a_global_error){
    //fprintf(stderr, "File %s - Line %i - Error after ThreadSynchronize %s : %s\n", currentFile, currentLine, errorMessage, oclErrorString(error));
    fprintf(stderr, "File %s - Line %i - Failed - %s : %d\n", 
	    currentFile, currentLine, errorMessage, p4a_global_error);
    exit(-1);
  }
  else {
    fprintf(stdout, "File %s - Line %i - Success - %s : %d\n", 
	    currentFile, currentLine, errorMessage, p4a_global_error);
  }
#endif 
}

inline void checkArgsInline(const char *kernel,...)
{
  if (!p4a_program) {
    // Length of <<kernel>> + 2*'.' + '/' 'c' + 'l' + '\0' (= +4 char)
    int size = strlen(kernel)+6;
    char* kernelFile = (char *)malloc(size);
    sprintf(kernelFile,"./%s.cl",kernel);
    PRINT_LOG("Program and Kernel creation from %s\n",kernelFile);
    
    size_t kernelLength;
    const char* cSourceCL = oclLoadProgSource(kernelFile,"// This kernel was generated for P4A\n",&kernelLength);
    if (cSourceCL == NULL)
      PRINT_LOG("source du program null\n");
    else
      PRINT_LOG("%s\n",cSourceCL);
    /*Create and compile the program : 1 for 1 kernel */
    p4a_program=clCreateProgramWithSource(p4a_context,1,
					  (const char **)&cSourceCL,
					  &kernelLength,
					  &p4a_global_error);
    toolTestExecMessage("clCreateProgramWithSource");
    p4a_global_error=clBuildProgram(p4a_program,0,NULL,NULL,NULL,NULL);
    toolTestExecMessage("clBuildProgram");
    p4a_kernel=clCreateKernel(p4a_program,kernel,&p4a_global_error);
    toolTestExecMessage("clCreateKernel");
  }
  va_list ap;
  va_start(ap, kernel);
  for (int i = 0;i < p4a_args_count;i++) {
    cl_mem m = va_arg(ap, cl_mem);
    p4a_global_error=clSetKernelArg(p4a_kernel,i,sizeof(cl_mem),(void *)&m);
    toolTestExecMessage("clSetKernelArg %d",i);
  }
  va_end(ap);
}

/** @defgroup P4A_cl_kernel_call Accelerator kernel call

    @{
*/

/** Parameters used to choose work_item allocation per work_group in OpenCL.

    This allow to choose the size of the blocks of work_item in 1,2 and 3D.

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
#define P4A_init_accel	                                                  \
  /* Get an OpenCL platform */						  \
  p4a_global_error=clGetPlatformIDs(1, &p4a_platform_id, NULL);           \
  toolTestExecMessage("clGetPlatformIDs");                                \
/* Get the devices,could be a collection of device */                     \
  p4a_global_error=clGetDeviceIDs(p4a_platform_id,                        \
			      CL_DEVICE_TYPE_GPU,                         \
			      1,                                          \
			      &p4a_device_id,                             \
			      NULL);                                      \
  toolTestExecMessage("clGetDeviceIDs");                                  \
  /* Create the context */                                                \
  p4a_context=clCreateContext(0, 1,&p4a_device_id, NULL, NULL,            \
			      &p4a_global_error);			  \
  toolTestExecMessage("clCreateContext");                                 \
  /* ... could query many device, we retain only the first one ... */     \
  /* Create a file allocated to the first device ...   */		  \
  p4a_queue=clCreateCommandQueue(p4a_context,p4a_device_id,0,             \
				 &p4a_global_error);			  \
  toolTestExecMessage("clCreateCommandQueue");


/** Release the hardware accelerator in CL

    Nothing to do
*/
#define P4A_release_accel                                               
    

/** @} */


/** @defgroup P4A_cl_time_measure Time execution measurement

    @{
*/

/** Start a timer on the accelerator in CL */
#define P4A_accel_timer_start 

/** @} */



/** A declaration attribute of a hardware-accelerated kernel in CL
    called from the GPU it-self
*/

#define P4A_accel_kernel inline void

/** A declaration attribute of a hardware-accelerated kernel called from
    the host in CL */
#define P4A_accel_kernel_wrapper __kernel void


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


/** @defgroup P4A_CL_memory_allocation_copy Memory allocation and copy for 
    CL acceleration

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
		       const void *host_address,
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
			    size_t d1_block_size, 
			    size_t d2_block_size, 
			    size_t d3_block_size,
			    size_t d1_offset,size_t d2_offset,size_t d3_offset,
			    void *host_address,
			    const void *accel_address);

/** Function for copying a 3D memory zone from the host to a compact memory
    zone in the hardware accelerator.
*/
void P4A_copy_to_accel_3d(size_t element_size,
			  size_t d1_size, size_t d2_size, size_t d3_size,
			  size_t d1_block_size, 
			  size_t d2_block_size, 
			  size_t d3_block_size,
			  size_t d1_offset, size_t d2_offset, size_t d3_offset,
			  const void *host_address,
			  void *accel_address);

/* @} */


/** @addtogroup P4A_cl_kernel_call

    @{
*/

/** Call a CL kernel on the accelerator.

    An API for full call control. For simpler usage: 
    @see P4A_call_accel_kernel_1d, 
    @see P4A_call_accel_kernel_2d, 
    @see P4A_call_accel_kernel_3d

    This transforms for example:

    P4A_call_accel_kernel((pips_accel_1, 1, pips_accel_dimBlock_1),
                          (*accel_imagein_re, *accel_imagein_im));

    into:

    do { pips_accel_1<<<1, pips_accel_dimBlock_1>>> (*accel_imagein_re, *accel_imagein_im); toolTestExecMessage ("P4A CL kernel execution failed", "init.cu", 58); } while (0);
*/
#define P4A_call_accel_kernel(context, parameters)			\
  do {									\
    P4A_skip_debug(P4A_dump_location());				\
    P4A_skip_debug(P4A_dump_message("Invoking %s with %s\n",	        \
				    #context,				\
				    #parameters));                      \
    P4A_call_accel_kernel_parameters parameters;			\
    toolTestExecMessage("P4A OpenCL kernel execution");		\
  } while (0)

/* @} */


/** CL kernel invocation.
*/
#define P4A_call_accel_kernel_context(kernel, ...)


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
  /* Define the number of thread per block: */				\
  cl_uint work_dim = 1;                                                 \
  size_t block_descriptor_name[] = {P4A_min((int) size,			\
					    (int) P4A_CL_ITEM_PER_GROUP_IN_1D)}; \
  /* Define the ceil-rounded number of needed blocks of threads: */	\
  size_t grid_descriptor_name[] = {(int)size)};                         \
  P4A_skip_debug(P4A_dump_grid_descriptor(grid_descriptor_name);)	\
  P4A_skip_debug(P4A_dump_block_descriptor(block_descriptor_name);)


/** Allocate the descriptors for a 2D set of thread with a simple
    strip-mining in each dimension for CL
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
    p4a_block_x = P4A_CL_ITEM_X_PER_GROUP_IN_2D;			\
    p4a_block_y = P4A_CL_ITEM_Y_PER_GROUP_IN_2D;			\
  }									\
  else {								\
    /* Allocate a maximum of threads alog X axis (the warp dimension) for \
       better average efficiency: */					\
    p4a_block_x = P4A_min((int) n_x_iter,				\
                          (int) P4A_CL_ITEM_MAX);			\
    p4a_block_y = P4A_min((int) n_y_iter,				\
                          P4A_CL_ITEM_MAX/p4a_block_x);		        \
  }									\
  cl_uint work_dim = 2;                                                 \
  size_t block_descriptor_name[]={(size_t)p4a_block_x,(size_t)p4a_block_y}; \
  /* Define the ceil-rounded number of needed blocks of threads: */	\
  size_t grid_descriptor_name[]={((size_t) n_x_iter),((size_t) n_y_iter)};\
  /*P4A_skip_debug(P4A_dump_grid_descriptor(grid_descriptor_name);)*/	\
  /*P4A_skip_debug(P4A_dump_block_descriptor(block_descriptor_name);)*/


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
    P4A_skip_debug(P4A_dump_message("Calling 1D kernel \"" #kernel "\" of size %d\n",	\
                   P4A_n_iter_0));					\
    P4A_create_1d_thread_descriptors(P4A_grid_descriptor,		\
				     P4A_block_descriptor,		\
				     P4A_n_iter_0);			\
    P4A_call_accel_kernel(clEnqueueNDRangeKernel,(p4a_queue,p4a_kernel,work_dim,NULL,&P4A_block_descriptor,&P4A_grid_descriptor,0,NULL,&p4a_event)); \
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
    checkArgsInline(kernel,__VA_ARGS__);				\
    P4A_skip_debug(P4A_dump_message("Calling 2D kernel \"" #kernel      \
                   "\" of size (%dx%d)\n",P4A_n_iter_0, P4A_n_iter_1)); \
    P4A_create_2d_thread_descriptors(P4A_grid_descriptor,		\
				     P4A_block_descriptor,		\
				     P4A_n_iter_0, P4A_n_iter_1);	\
    P4A_call_accel_kernel(clEnqueueNDRangeKernel,                       \
                          (p4a_queue,p4a_kernel,work_dim,NULL,          \
			   &P4A_block_descriptor,&P4A_grid_descriptor,  \
			   0,NULL,&p4a_event));				\
  } while (0)

/** @} */

#endif //P4A_ACCEL_CL_H
