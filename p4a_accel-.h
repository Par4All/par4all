/** @file

    API of Par4All C to OpenMP

    This file is an implementation of the C to CUDA Par4All API.

    Funded by the FREIA (French ANR), TransMedi\@ (French Pôle de
    Compétitivité Images and Network) and SCALOPES (Artemis European
    Project project)

    "mailto:Ronan.Keryell@hpc-project.com"
*/

/* License BSD */

#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

struct timeval p4a_time_begin, p4a_time_end;

struct {
  int x;
  int y;
  int z;
} threadIdx;


/** @defgroup P4A_OpenMP_time_measure Time execution measurement

    @{
*/

/** Start a timer on the accelerator */
#define P4A_ACCEL_TIMER_START gettimeofday(&p4a_time_begin, NULL)

double P4A_ACCEL_TIMER_STOP_AND_FLOAT_MEASURE();

/** Stop a timer on the accelerator and get double ms time */
/*
double P4A_ACCEL_TIMER_STOP_AND_FLOAT_MEASURE() {
  double run_time;
  gettimeofday(&p4a_time_end, NULL);
  // Take care of the non-associativity in floating point :-) 
  run_time = (p4a_time_end.tv_sec - p4a_time_begin.tv_sec)
    + (p4a_time_end.tv_usec - p4a_time_begin.tv_usec)*1e-6;
  return run_time;
}
*/

/** @} */


/** @defgroup P4A_init Initialization of P4A C to OpenMP

    @{
*/

/** Associate the program to the accelerator

    Initialized the use of the hardware accelerator

    Nothing to do
*/
#define P4A_INIT_ACCEL

/** Release the hardware accelerator as OpenMP

    Nothing to do
 */
#define P4A_RELEASE_ACCEL

/** @} */


/** A declaration attribute of a hardware-accelerated kernel

    Nothing by default
*/
#define P4A_ACCEL_KERNEL

/** A declaration attribute of a hardware-accelerated kernel called from
    the host 

    Nothing by default
*/

#define P4A_ACCEL_KERNEL_WRAPPER 

/* Use thread-local storage to pass iteration index ? */

/** Get the coordinate of the virtual processor in X (first) dimension */
//#define P4A_VP_X 0
#define P4A_VP_X threadIdx.x


/** Get the coordinate of the virtual processor in Y (second) dimension */
#define P4A_VP_Y threadIdx.y

/** Get the coordinate of the virtual processor in Z (second) dimension */
#define P4A_VP_Z threadIdx.z


/** @defgroup P4A_memory_allocation_copy Memory allocation and copy

    @{
*/

/** Allocate memory on the hardware accelerator

    For OpenMP it is on the host too

    @param[out] address is the address of a variable that is updated by
    this macro to contains the address of the allocated memory block

    @param[in] size is the size to allocate in bytes
*/
#define P4A_ACCEL_MALLOC(address, size)		\
  *(void **)address = malloc(size)


/** Free memory on the hardware accelerator

    It is on the host too

    @param[in] address is the address of a previously allocated memory zone on
    the hardware accelerator
*/
#define P4A_ACCEL_FREE(address)			\
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
#define P4A_COPY_TO_ACCEL(host_address, accel_address, size)		\
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
#define P4A_COPY_FROM_ACCEL(host_address, accel_address, size)	\
  /* Easy implementation since memcpy is symetrical :-) */	\
  P4A_COPY_TO_ACCEL(accel_address, host_address, size)


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
#define P4A_CALL_ACCEL_KERNEL(context, parameters)			

/* @} */


/** CUDA kernel invocation.

    Generate something like "kernel<<<block_dimension,thread_dimension>>>"
*/
#define P4A_CALL_ACCEL_KERNEL_CONTEXT(kernel, ...)	


/** Add CUDA kernel parameters for invocation.

    Simply add them in parenthesis.  Well, this could be done without
    variadic arguments... Just for fun. :-)
*/
#define P4A_CALL_ACCEL_KERNEL_PARAMETERS(...)	


/** Creation of block and thread descriptors */

/** Allocate the descriptors for a linear set of thread with a
    simple strip-mining
*/
#define P4A_CREATE_1D_THREAD_DESCRIPTORS(block_descriptor_name,		\
					 grid_descriptor_name,		\
					 size)				


/** Allocate the descriptors for a 2D set of thread with a simple
    strip-mining in each dimension
*/
#define P4A_CREATE_2D_THREAD_DESCRIPTORS(block_descriptor_name,		\
					 grid_descriptor_name,		\
					 n_x_iter, n_y_iter)		


/** Dump a CUDA dim3 descriptor with an introduction message */
#define P4A_DUMP_DESCRIPTOR(message, descriptor_name)			


/** Dump a CUDA dim3 block descriptor */
#define P4A_DUMP_BLOCK_DESCRIPTOR(descriptor_name)			


/** Dump a CUDA dim3 grid descriptor */
#define P4A_DUMP_GRID_DESCRIPTOR(descriptor_name)		


/** @addtogroup P4A_cuda_kernel_call

    @{
*/

/** Call a kernel in a 1-dimension parallel loop

    Translate it into an OpenMP parallel loop

    @param[in] kernel to call

    @param[in] size is the number of iterations

    @param ... the following parameters are given to the kernel
*/
#define P4A_CALL_ACCEL_KERNEL_1D(kernel, size, ...)		\
  int P4A_index_x;                                              \
  for(int P4A_index_x = 0; P4A_index_x < size; P4A_index_x++) {	\
    const int P4A_index_y = 0;					\
    const int P4A_index_z = 0;					\
    kernel(__VA_ARGS__);					\
  }


/** Call a kernel in a 2-dimension parallel loop

    Translate it into an OpenMP parallel loop-nest

    @param[in] kernel to call

    @param[in] n_x_iter is the number of iterations in the first dimension

    @param[in] n_y_iter is the number of iterations in the first dimension

    @param ... following parameters are given to the kernel
*/
#define P4A_CALL_ACCEL_KERNEL_2D(kernel, n_x_iter, n_y_iter, ...)	\
  int P4A_index_x,P4A_index_y;                                          \
  for(P4A_index_x = 0; P4A_index_x < n_x_iter; P4A_index_x++) {	        \
    for(P4A_index_y = 0; P4A_index_y < n_y_iter; P4A_index_y++) {	\
      const int P4A_index_z = 0;					\
      threadIdx.x = P4A_index_x;					\
      threadIdx.y = P4A_index_y;					\
      threadIdx.z = P4A_index_z;                                        \
     kernel(__VA_ARGS__);						\
    }									\
  }

/** @} */
