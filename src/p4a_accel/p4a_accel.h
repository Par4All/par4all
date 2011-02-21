/** @file

    API of Par4All C for accelerator

    This files contains 
    - prototypes of general functions defined in p4a_accel.c
    - includes the header .h in function of the hardware

    @mainpage

    The Par4All API leverages the code to program an hardware accelerator
    with some helper functions.

    It allows simpler and more portable code to be generated with PIPS,
    but it can also be used to ease heterogeneous parallel manual
    programming.

    Should target Ter\@pix and SCALOPES accelerators, CUDA and OpenCL.

    Funded by the FREIA (French ANR), TransMedi\@ (French Pôle de
    Compétitivité Images and Network) and SCALOPES (Artemis European
    Project project), with the support of the MODENA project (French Pôle de
    Compétitivité Mer Bretagne)

    "mailto:Stephanie.Even@enstb.org"
    "mailto:Ronan.Keryell@hpc-project.com"
*/

// License BSD

#ifndef P4A_ACCEL_H
#define P4A_ACCEL_H

/*
#ifdef __cplusplus
extern "C" {
#endif
*/

// For size_t
#include <stddef.h>

/** @defgroup P4A_time_measure Time measurement

    The time is measured with the call to two markers: 
    -# the P4A_accel_timer_start macro to begin;
    -# P4A_accel_timer_stop_and_float_measure() to end.

    @{
*/

/** Prototype of the function that ends the timer in OpenMP, CUDA and
    OpenCL.
 */
extern double P4A_accel_timer_stop_and_float_measure();

/** @}
*/

#if defined(P4A_ACCEL_CUDA) && defined(P4A_ACCEL_OPENMP) 
#error "You cannot have both P4A_ACCEL_CUDA and P4A_ACCEL_OPENMP defined, yet"
#endif

#if defined(P4A_ACCEL_CUDA) && defined(P4A_ACCEL_OPENCL) 
#error "You cannot have both P4A_ACCEL_CUDA and P4A_ACCEL_OPENCL defined, yet"
#endif

#if defined(P4A_ACCEL_OPENCL) && defined(P4A_ACCEL_OPENMP) 
#error "You cannot have both P4A_ACCEL_OPENCL and P4A_ACCEL_OPENMP defined, yet"
#endif

/* Some common function prototypes. */

/** @defgroup P4A_memory_allocation_copy Memory allocation and copy 

    @{
*/

/** Prototype for allocating memory on the hardware accelerator.

    @param[out] address is the address of a variable that is updated by
    this macro to contains the address of the allocated memory block

    @param[in] size is the size to allocate in bytes
*/
void P4A_accel_malloc(void **address, size_t size);


/** Prototype for freeing memory on the hardware accelerator/

    @param[in] address points to a previously allocated memory zone for
    the hardware accelerator
*/
void P4A_accel_free(void *address);

/** Prototype for copying a scalar from the host to a memory zone in the
    hardware accelerator.

    It's a wrapper around memcpy, CudaMemCopy* in CUDA or clEnqueueWriteBuffer 
    in OpenCL.
 
    Do not change the place of the pointers in the API in opposition
    to CUDA. The host address is always before the accel address...

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


/** Prototype for copying a scalar from the hardware accelerator memory to
    the host.

    It's a wrapper around memcpy, CudaMemCopy* in CUDA or clEnqueueReadBuffer 
    in OpenCL.
 
    Do not change the place of the pointers in the API, in opposition
    to CUDA. The host address is always first...
    
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


/** Prototype for copying a 1D memory zone from the host to a compact
    memory zone in the hardware accelerator.

    It's a wrapper around memcpy, CudaMemCopy* in CUDA or clEnqueueWriteBuffer 
    in OpenCL.

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


/** Prototype for copying memory from the hardware accelerator to a 1D
    array in the host.

    It's a wrapper around memcpy, CudaMemCopy* in CUDA or clEnqueueReadBuffer 
    in OpenCL.
 
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
*/
void P4A_copy_from_accel_1d(size_t element_size,
			    size_t d1_size,
			    size_t d1_block_size,
			    size_t d1_offset,
			    void *host_address,
			    const void *accel_address);


/** Prototype for copying a 2D memory zone from the host to a compact
    memory zone in the hardware accelerator.
*/
void P4A_copy_to_accel_2d(size_t element_size,
			  size_t d1_size, size_t d2_size,
			  size_t d1_block_size, size_t d2_block_size,
			  size_t d1_offset,   size_t d2_offset,
			  const void *host_address,
			  void *accel_address);


/** Prototype for copying memory from the hardware accelerator to a 2D
    array in the host.
*/
void P4A_copy_from_accel_2d(size_t element_size,
			    size_t d1_size, size_t d2_size,
			    size_t d1_block_size, size_t d2_block_size,
			    size_t d1_offset, size_t d2_offset,
			    void *host_address,
			    const void *accel_address);


/** Prototype for copying a 3D memory zone from the host to a compact
    memory zone in the hardware accelerator.
*/
void P4A_copy_to_accel_3d(size_t element_size,
			  size_t d1_size, size_t d2_size, size_t d3_size,
			  size_t d1_block_size, size_t d2_block_size, size_t d3_block_size,
			  size_t d1_offset,   size_t d2_offset, size_t d3_offset,
			  const void *host_address,
			  void *accel_address);


/** Prototype for copying memory from the hardware accelerator to a 3D
    array in the host.
*/
void P4A_copy_from_accel_3d(size_t element_size,
			    size_t d1_size, size_t d2_size, size_t d3_size,
			    size_t d1_block_size, size_t d2_block_size, size_t d3_block_size,
			    size_t d1_offset, size_t d2_offset, size_t d3_offset,
			    void *host_address,
			    const void *accel_address);
/* @} */

/** @defgroup P4A_log Debug messages and errors tracking.
    
    @{
*/

/** Formats the standard MACROS  __FILE__ and __LINE__ for message print.
 */
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#define AT __FILE__ ":" TOSTRING(__LINE__)



/** Prototype for copying memory from a 4D array in the host to the hardware
    accelerator.
*/
void P4A_copy_to_accel_4d(size_t element_size,
          size_t d1_size, size_t d2_size, size_t d3_size, size_t d4_size,
          size_t d1_block_size, size_t d2_block_size, size_t d3_block_size, size_t d4_block_size,
          size_t d1_offset, size_t d2_offset, size_t d3_offset, size_t d4_offset,
          const void *host_address,
          void *accel_address);


/** Prototype for copying memory from the hardware accelerator to a 4D
    array in the host.
*/
void P4A_copy_from_accel_4d(size_t element_size,
          size_t d1_size, size_t d2_size, size_t d3_size, size_t d4_size,
          size_t d1_block_size, size_t d2_block_size, size_t d3_block_size, size_t d4_block_size,
          size_t d1_offset, size_t d2_offset, size_t d3_offset, size_t d4_offset,
          void *host_address,
          const void *accel_address);



/**
 * This function return a pointer on the area in the GPU memory corresponding to
 * an area in the host memory. If the mapping doesn't exist, an area is
 * allocated in the GPU memory and the mapping is created
 *
 * @params host_ptr is the pointer in the host memory
 * @params size is the size of the area, in case we need to allocate it
 * @return a pointer in the GPU memory corresponding to host_ptr
 */
void * P4A_runtime_host_ptr_to_accel_ptr(void *host_ptr, size_t size);


/**
 * This function copy "size" bytes from "host_ptr" to the corresponding area in
 * the GPU memory. The memory area can be allocated if there's none existing
 *
 * @params host_ptr is the pointer in the host memory
 * @params size is the size of the area
 * @return nothing
 */
void P4A_runtime_copy_to_accel(void *host_ptr, size_t size /* in bytes */);



/**
 * This function copy "size" bytes to "host_ptr" from the corresponding area in
 * the GPU memory. An abort is raised if no mapping if found.
 *
 * @params host_ptr is the pointer in the host memory
 * @params size is the size of the area
 * @return nothing
 */
void P4A_runtime_copy_from_accel(void *host_ptr, size_t size /* in bytes */);



/** A macro to enable or skip debug instructions

    Just define P4A_DEBUG to have debug information at runtime

    @param debug_stuff is some text that is included texto if P4A_DEBUG is
    defined
*/

#ifdef P4A_DEBUG
#define P4A_skip_debug(debug_stuff) debug_stuff
#else
#define P4A_skip_debug(debug_stuff)
#endif

#include <stdio.h>

/** Output a debug message à la printf */
#define P4A_dump_message(...)			\
  fprintf(stderr, " P4A: " __VA_ARGS__)

/** Output where we are */
#define P4A_dump_location()					\
  fprintf(stderr, " P4A: %s, function %s\n",AT,__func__)

/** @} */


/** @defgroup P4A_util Useful functions
    
    @{
*/

/** A macro to compute an minimum value of 2 values

    Since it is a macro, beware of side effects...
*/
#define P4A_min(a, b) ((a) > (b) ? (b) : (a))

/** @} */

#ifdef P4A_ACCEL_CUDA
#include <p4a_accel-CUDA.h>
#else
#ifdef P4A_ACCEL_OPENMP
#include <p4a_accel-OpenMP.h>
#else
#ifdef P4A_ACCEL_OPENCL
#include <p4a_accel-OpenCL.h>
#else
#error "You have to define either P4A_ACCEL_CUDA, P4A_ACCEL_OPENMP or P4A_ACCEL_OPENCL"
#endif
#endif
#endif

  /*
#ifdef __cplusplus
}
#endif
  */

#endif //P4A_ACCEL_H
