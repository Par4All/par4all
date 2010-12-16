/** @file

    API of Par4All C to OpenMP

    This file is an implementation of the C to accelerator Par4All API.

    Funded by the FREIA (French ANR), TransMedi\@ (French Pôle de
    Compétitivité Images and Network) and SCALOPES (Artemis European
    Project project), with participation of MODENA (French Pôle de
    Compétitivité Mer Bretagne)

    "mailto:Stephanie.Even@enstb.org"
    "mailto:Ronan.Keryell@hpc-project.com"
*/

#include <p4a_accel.h>


#ifdef P4A_ACCEL_OPENMP

/** Stop a timer on the accelerator and get double ms time

    @addtogroup P4A_OpenMP_time_measure
*/

double P4A_accel_timer_stop_and_float_measure() 
{
#ifdef P4A_PROFILING
  double run_time;
  gettimeofday(&p4a_time_end, NULL);
  /* Take care of the non-associativity in floating point :-) */
  run_time = (p4a_time_end.tv_sec - p4a_time_begin.tv_sec)
    + (p4a_time_end.tv_usec - p4a_time_begin.tv_usec)*1e-6;
  return run_time;
#else
  return 0;
#endif
}


/** This is a global variable used to simulate P4A virtual processor
    coordinates in OpenMP because we need to pass a local variable to a
    function without passing it in the arguments.

    Use thead local storage to have it local to each OpenMP thread.

    With __thread, it looks like this declaration cannot be repeated in
    the .h without any extern.
 */
__thread int P4A_vp_coordinate[P4A_vp_dim_max];


/** @defgroup P4A_OpenMP_memory_allocation_copy Memory allocation and copy for OpenMP simulation

    @{
*/

/** Allocate memory on the hardware accelerator in OpenMP emulation.

    For OpenMP it is on the host too.

    @param[out] address is the address of a variable that is updated by
    this macro to contains the address of the allocated memory block

    @param[in] size is the size to allocate in bytes
*/
void P4A_accel_malloc(void **address, size_t size) 
{
  *address = malloc(size);
}


/** Free memory on the hardware accelerator in OpenMP emulation.

    It is on the host too

    @param[in] address points to a previously allocated memory zone for
    the hardware accelerator
*/
void P4A_accel_free(void *address) 
{
  free(address);
}

/** Copy a scalar from the hardware accelerator to the host in OpenMP
 emulation.

 Since it is an OpenMP implementation, use standard memory copy
 operations

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
			 const void *accel_address) 
{
  /* We can use memcpy() since we are sure there is no overlap */
  memcpy(host_address, accel_address, element_size);
}

/** Copy a scalar from the host to the hardware accelerator in OpenMP
 emulation.

 Since it is an OpenMP implementation, use standard memory copy
 operations

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
		       void *accel_address) 
{
  /* We can use memcpy() since we are sure there is no overlap */
  memcpy(accel_address, host_address, element_size);
}

/** Function for copying memory from the hardware accelerator to a 1D array in
 the host.

 This function could be quite simpler but is designed by symmetry with
 other functions.

 @param[in] element_size is the size of one element of the array in
 byte

 @param[in] d1_size is the number of elements in the array. It is not
 used but here for symmetry with functions of higher dimensionality

 @param[in] d1_block_size is the number of element to transfer

 @param[in] d1_offset is element order to start the transfer from  (host side)

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
			    const void *accel_address) 
{
  // Compute the destination address on host side
  char * cdest = d1_offset*element_size + (char *)host_address;
  const char * csrc = (char*)accel_address;
  // Copy element by element
  for(size_t i = 0; i < d1_block_size*element_size; i++)
    cdest[i] = csrc[i];
}

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
			  void *accel_address) 
{
  // Compute the destination address on host side
  const char * csrc = d1_offset*element_size + (char *)host_address;
  char * cdest = (char*)accel_address;

  // Copy element by element
  for(size_t i = 0; i < d1_block_size*element_size; i++)
    cdest[i] = csrc[i];
}

/** Function for copying memory from the hardware accelerator to a 2D array in
 the host.
 */
void P4A_copy_from_accel_2d(size_t element_size,
			    size_t d1_size, 
			    size_t d2_size,
			    size_t d1_block_size, 
			    size_t d2_block_size,
			    size_t d1_offset, 
			    size_t d2_offset,
			    void *host_address,
			    const void *accel_address) 
{
  // Compute the destination address for the first rown on dim1 on host side
  char * cdest = d2_offset*element_size + (char*)host_address;
  const char * csrc = (char*)accel_address;

  // Copy element by element
  for(size_t i = 0; i < d1_block_size; i++)
    for(size_t j = 0; j < d2_block_size*element_size; j++)
      cdest[(i + d1_offset)*element_size*d2_size + j] =
          csrc[i*element_size*d2_block_size + j];
}

/** Function for copying a 2D memory zone from the host to a compact memory
 zone in the hardware accelerator.
 */
void P4A_copy_to_accel_2d(size_t element_size,
			  size_t d1_size, 
			  size_t d2_size,
			  size_t d1_block_size, 
			  size_t d2_block_size,
			  size_t d1_offset, 
			  size_t d2_offset,
			  const void *host_address,
			  void *accel_address) 
{
  char * cdest = (char *)accel_address;
  const char * csrc = d2_offset*element_size + (char *)host_address;
  for(size_t i = 0; i < d1_block_size; i++)
    for(size_t j = 0; j < d2_block_size*element_size; j++)
      cdest[i*element_size*d2_block_size + j] = csrc[(i + d1_offset)*element_size*d2_size + j];
}

/** Function for copying memory from the hardware accelerator to a 3D array in
    the host.
*/
void P4A_copy_from_accel_3d(size_t element_size,
			    size_t d1_size, 
			    size_t d2_size, 
			    size_t d3_size,
			    size_t d1_block_size, 
			    size_t d2_block_size, 
			    size_t d3_block_size,
			    size_t d1_offset, 
			    size_t d2_offset, 
			    size_t d3_offset,
			    void *host_address,
			    const void *accel_address) 
{
  char * cdest = d3_offset*element_size + (char*)host_address;
  const char * csrc = (char*)accel_address;
  for(size_t i = 0; i < d1_block_size; i++)
    for(size_t j = 0; j < d2_block_size; j++)
      for(size_t k = 0; k < d3_block_size*element_size; k++)
	cdest[((i + d1_offset)*d2_block_size + j + d2_offset)*element_size*d3_size + k] =
	  csrc[(i*d2_block_size + j)*d3_block_size*element_size + k];
}


/** Function for copying a 3D memory zone from the host to a compact memory
    zone in the hardware accelerator.
*/
void P4A_copy_to_accel_3d(size_t element_size,
			  size_t d1_size, 
			  size_t d2_size, 
			  size_t d3_size,
			  size_t d1_block_size, 
			  size_t d2_block_size, 
			  size_t d3_block_size,
			  size_t d1_offset,   
			  size_t d2_offset, 
			  size_t d3_offset,
			  const void *host_address,
			  void *accel_address) 
{
  char * cdest = (char *)accel_address;
  const char * csrc = d3_offset*element_size + (char *)host_address;
  for(size_t i = 0; i < d1_block_size; i++)
    for(size_t j = 0; j < d2_block_size; j++)
      for(size_t k = 0; k < d3_block_size*element_size; k++)
	cdest[(i*d2_block_size + j)*d3_block_size*element_size + k] =
	  csrc[((i + d1_offset)*d2_block_size + j + d2_offset)*element_size*d3_size + k];
}

/* @} */
#endif // P4A_ACCEL_OPENMP

#ifdef P4A_ACCEL_CUDA

//#include <cutil_inline.h>

/** To do basic time measure. Do not nest... */

cudaEvent_t p4a_start_event, p4a_stop_event;

/** Stop a timer on the accelerator and get float time in second

 @addtogroup P4A_cuda_time_measure
 */

double P4A_accel_timer_stop_and_float_measure() 
{
#ifdef P4A_PROFILING
  float execution_time;
  P4A_test_execution(cudaEventRecord(p4a_stop_event, 0));
  P4A_test_execution(cudaEventSynchronize(p4a_stop_event));
  /* Get the time in ms: */
  P4A_test_execution(cudaEventElapsedTime(&execution_time,
					  p4a_start_event,
					  p4a_stop_event));
  /* Return the time in second: */
  return execution_time*1e-3;
#else
  return 0;
#endif
}



/** Allocate memory on the hardware accelerator in Cuda mode.

    @param[out] address is the address of a variable that is updated by
    this function to contains the address of the allocated memory block

    @param[in] size is the size to allocate in bytes
*/
void P4A_accel_malloc(void **address, size_t size) 
{
  P4A_test_execution(cudaMalloc(address, size));
}


/** Free memory on the hardware accelerator in Cuda mode

    @param[in] address points to a previously allocated memory zone for
    the hardware accelerator
*/
void P4A_accel_free(void *address) 
{
  P4A_test_execution(cudaFree(address));
}

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
			 const void *accel_address) 
{
  cudaMemcpy(host_address,accel_address,element_size,cudaMemcpyDeviceToHost);
}

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
		       void *accel_address) 
{
  cudaMemcpy(accel_address,host_address,element_size,cudaMemcpyHostToDevice);
}

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
			    const void *accel_address) 
{
  char * cdest = d1_offset*element_size + (char *)host_address;
  cudaMemcpy(cdest,accel_address,d1_block_size*element_size,cudaMemcpyDeviceToHost);
}

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
			  void *accel_address) 
{
  const char * csrc = d1_offset*element_size + (char *)host_address;
  cudaMemcpy(accel_address,csrc,d1_block_size*element_size,cudaMemcpyHostToDevice);
}

/** Function for copying memory from the hardware accelerator to a 2D array in
 the host.
 */
void P4A_copy_from_accel_2d(size_t element_size,
			    size_t d1_size, 
			    size_t d2_size,
			    size_t d1_block_size, 
			    size_t d2_block_size,
			    size_t d1_offset, 
			    size_t d2_offset,
			    void *host_address,
			    const void *accel_address) 
{
  if(d2_size==d2_block_size ) { // d2 is fully transfered ? We can optimize :-)
    // We transfer all in one shot !
    P4A_copy_from_accel_1d(element_size,
                           d2_size * d1_size,
                           d2_size * d1_block_size,
                           d1_offset*d2_size+d2_offset,
                           host_address,
                           accel_address);
  } else { // Transfer row by row !
    // Compute the adress of the begining of the first row to transfer
    char * host_row = (char*)host_address + d1_offset*d2_size*element_size;
    char * accel_row = (char*)accel_address;

    // Will transfert "d1_block_size" rows
    for(size_t i = 0; i < d1_block_size; i++) {
      P4A_copy_from_accel_1d(element_size,
                             d2_size,
                             d2_block_size,
                             d2_offset,
                             host_row,
                             accel_row);

      // Comput addresses for next row
      host_row += d2_size*element_size;
      accel_row += d2_block_size*element_size;
    }
  }
}

/** Function for copying a 2D memory zone from the host to a compact memory
 zone in the hardware accelerator.
 */
void P4A_copy_to_accel_2d(size_t element_size,
			  size_t d1_size, 
			  size_t d2_size,
			  size_t d1_block_size, 
			  size_t d2_block_size,
			  size_t d1_offset, 
			  size_t d2_offset,
			  const void *host_address,
			  void *accel_address) 
{
  if(d2_size==d2_block_size ) { // d2 is fully transfered ? We can optimize :-)
    // We transfer all in one shot !
    P4A_copy_to_accel_1d(element_size,
                         d2_size * d1_size,
                         d2_size * d1_block_size,
                         d1_offset*d2_size+d2_offset,
                         host_address,
                         accel_address);
  } else { // Transfer row by row !
    // Compute the adress of the begining of the first row to transfer
    char * host_row = (char*)host_address + d1_offset*d2_size*element_size;
    char * accel_row = (char*)accel_address;

    // Will transfert "d1_block_size" rows
    for(size_t i = 0; i < d1_block_size; i++) {
      P4A_copy_to_accel_1d(element_size,
                           d2_size,
                           d2_block_size,
                           d2_offset,
                           host_row,
                           accel_row);

      // Comput addresses for next row
      host_row += d2_size*element_size;
      accel_row += d2_block_size*element_size;
    }
  }

}

/** Function for copying memory from the hardware accelerator to a 3D array in
    the host.
*/
void P4A_copy_from_accel_3d(size_t element_size,
			    size_t d1_size, 
			    size_t d2_size, 
			    size_t d3_size,
			    size_t d1_block_size, 
			    size_t d2_block_size, 
			    size_t d3_block_size,
			    size_t d1_offset, 
			    size_t d2_offset, 
			    size_t d3_offset,
			    void *host_address,
			    const void *accel_address) 
{
  if(d3_size==d3_block_size ) { // d3 is fully transfered ? We can optimize :-)
    // We transfer all in one shot !
    P4A_copy_from_accel_2d(element_size,
                           d1_size,d3_size * d2_size,
                           d1_block_size, d3_size * d2_block_size,
                           d1_offset, d2_offset*d3_size+d3_offset,
                           host_address,
                           accel_address);
  } else { // Transfer row by row !
    // Compute the adress of the begining of the first row to transfer
    char * host_row = (char*)host_address + d1_offset*d2_size*d3_size*element_size;
    char * accel_row = (char*)accel_address;

    // Will transfert "d1_block_size" rows
    for(size_t i = 0; i < d1_block_size; i++) {
      P4A_copy_from_accel_2d(element_size,
                             d2_size,d3_size,
                             d2_block_size, d3_block_size,
                             d2_offset, d3_offset,
                             host_row,
                             accel_row);

      // Comput addresses for next row
      host_row += d3_size*element_size;
      accel_row += d3_block_size*element_size;
    }
  }
}


/** Function for copying a 3D memory zone from the host to a compact memory
    zone in the hardware accelerator.
*/
void P4A_copy_to_accel_3d(size_t element_size,
			  size_t d1_size, 
			  size_t d2_size, 
			  size_t d3_size,
			  size_t d1_block_size, 
			  size_t d2_block_size, 
			  size_t d3_block_size,
			  size_t d1_offset,   
			  size_t d2_offset, 
			  size_t d3_offset,
			  const void *host_address,
			  void *accel_address) 
{
  // d3 is fully transfered ? We can optimize :-)
  if (d3_size == d3_block_size ) { 
    // We transfer all in one shot !
    P4A_copy_to_accel_2d(element_size,
                         d1_size,d3_size * d2_size,
                         d1_block_size, d3_size * d2_block_size,
                         d1_offset, d2_offset*d3_size+d3_offset,
                         host_address,
                         accel_address);
  } 
  else { // Transfer row by row !
    // Compute the adress of the begining of the first row to transfer
    char * host_row = (char*)host_address + d1_offset*d2_size*d3_size*element_size;
    char * accel_row = (char*)accel_address;
    
    // Will transfert "d1_block_size" rows
    for (size_t i = 0; i < d1_block_size; i++) {
      P4A_copy_to_accel_2d(element_size,
                           d2_size,d3_size,
                           d2_block_size, d3_block_size,
                           d2_offset, d3_offset,
                           host_row,
                           accel_row);

      // Comput addresses for next row
      host_row += d3_size*element_size;
      accel_row += d3_block_size*element_size;
    }
  }
}

/** Function for copying memory from the hardware accelerator to a 4D array in
    the host.
*/
void P4A_copy_from_accel_4d(size_t element_size,
			    size_t d1_size, 
			    size_t d2_size, 
			    size_t d3_size, 
			    size_t d4_size,
			    size_t d1_block_size, 
			    size_t d2_block_size, 
			    size_t d3_block_size, 
			    size_t d4_block_size,
			    size_t d1_offset, 
			    size_t d2_offset, 
			    size_t d3_offset, 
			    size_t d4_offset,
			    void *host_address,
			    const void *accel_address) 
{
  // d4 is fully transfered ? We can optimize :-)
  if (d4_size == d4_block_size ) { 
    // We transfer all in one shot !
    P4A_copy_from_accel_3d(element_size,
			   d1_size,d2_size,d3_size * d4_size,
			   d1_block_size, d2_block_size, 
			   d3_block_size * d4_size,
			   d1_offset, d2_offset, 
			   d3_offset*d4_size+d4_offset,
			   host_address,
			   accel_address);
  } 
  else { // Transfer row by row !
    // Compute the adress of the begining of the first row to transfer
    char * host_row = (char*)host_address + d1_offset*d2_size*d3_size*d4_size*element_size;
    char * accel_row = (char*)accel_address;

    // Will transfert "d1_block_size" rows
    for (size_t i = 0; i < d1_block_size; i++) {
      P4A_copy_from_accel_3d(element_size,
                           d2_size,d3_size,d4_size,
                           d2_block_size, d3_block_size, d4_block_size,
                           d2_offset, d3_offset, d4_offset,
                           host_row,
                           accel_row);

      // Comput addresses for next row
      host_row += d4_size*element_size;
      accel_row += d4_block_size*element_size;
    }
  }
}

/** Function for copying a 4D memory zone from the host to a compact memory
    zone in the hardware accelerator.
*/
void P4A_copy_to_accel_4d(size_t element_size,
			  size_t d1_size, 
			  size_t d2_size, 
			  size_t d3_size, 
			  size_t d4_size,
			  size_t d1_block_size, 
			  size_t d2_block_size, 
			  size_t d3_block_size, 
			  size_t d4_block_size,
			  size_t d1_offset, 
			  size_t d2_offset, 
			  size_t d3_offset, 
			  size_t d4_offset,
			  const void *host_address,
			  void *accel_address) 
{
  // d4 is fully transfered ? We can optimize :-)
  if (d4_size == d4_block_size ) { 
    // We transfer all in one shot !
    P4A_copy_to_accel_3d(element_size,
                         d1_size,d2_size,d3_size * d4_size,
                         d1_block_size, d2_block_size, d3_block_size * d4_size,
                         d1_offset, d2_offset, d3_offset*d4_size+d4_offset,
                         host_address,
                         accel_address);
  } 
  else { // Transfer row by row !
    // Compute the adress of the begining of the first row to transfer
    char * host_row = (char*)host_address + d1_offset*d2_size*d3_size*d4_size*element_size;
    char * accel_row = (char*)accel_address;
    
    // Will transfert "d1_block_size" rows
    for (size_t i = 0; i < d1_block_size; i++) {
      P4A_copy_to_accel_3d(element_size,
                           d2_size,d3_size,d4_size,
                           d2_block_size, d3_block_size, d4_block_size,
                           d2_offset, d3_offset, d4_offset,
                           host_row,
                           accel_row);

      // Comput addresses for next row
      host_row += d4_size*element_size;
      accel_row += d4_block_size*element_size;
    }
  }
}


#endif //P4A_ACCEL_CUDA

#ifdef P4A_ACCEL_OPENCL

#include <string.h>

/** @author Stéphanie Even

    Aside the procedure defined for CUDA or OpenMP, OpenCL needs
    loading the kernel from a source file.

    Procedure to manage a kernel list have thus been created for the C version.
    In C++, this a map.

    C and C++ versions are implemented because of the argument passing.
 */

#ifdef P4A_PROFILING
cl_command_queue_properties p4a_queue_properties = CL_QUEUE_PROFILING_ENABLE;
#else
cl_command_queue_properties p4a_queue_properties = 0;
#endif

bool p4a_time_tag=false;
double p4a_time_execution = 0.;
double p4a_time_copy = 0.;
cl_event p4a_event_execution=NULL, p4a_event_copy=NULL;
cl_int p4a_global_error=0;
cl_context p4a_context = NULL;
cl_command_queue p4a_queue = NULL;
cl_program p4a_program = NULL;  
cl_kernel p4a_kernel = NULL;  

#ifdef __cplusplus
std::map<std::string,struct p4a_cl_kernel *> p4a_kernels_map;
#else
struct p4a_cl_kernel *p4a_kernels=NULL;
// Only used when creating the kernel list
struct p4a_cl_kernel *last_kernel=NULL;

// The C++ equivalent function is in the p4a_accel-OpenCL.h file
void p4a_setArguments(int i,char *s,size_t size,void * ref_arg) {
  //printf("%s : size %lu, rang %d\n",s,size,i);
  p4a_global_error = clSetKernelArg(p4a_kernel,i,size,ref_arg);
  P4A_test_execution_with_message("clSetKernelArg");
}

/** @author : Stéphanie Even

    Creation of a new kernel, initialisation and insertion in the kernel list.
 */
struct p4a_cl_kernel* new_p4a_kernel(const char *kernel)
{
  struct p4a_cl_kernel *new_kernel = (struct p4a_cl_kernel *)malloc(sizeof(struct p4a_cl_kernel));

  if (p4a_kernels == NULL) {
    //P4A_log("Begin of the kernel list\n");
    last_kernel = p4a_kernels = new_kernel;
  }
  else {
    //P4A_log("Follows the kernel list\n");
    last_kernel->next = new_kernel;
    last_kernel = new_kernel;
  }
  new_kernel->next = NULL;
  
  new_kernel->name = (char *)strdup(kernel);
  new_kernel->kernel = NULL;
  //new_kernel->n_args = 0;
  //new_kernel->args = NULL;
  //new_kernel->current = NULL;

  char* kernelFile;
  asprintf(&kernelFile,"./%s.cl",kernel);
  new_kernel->file_name = (char *)strdup(kernelFile);

  //P4A_log("A new kernel has been created and initialized\n");
  return new_kernel;
}

/** @author : Stéphanie Even

    Search if the <kernel> is already in the list.
 */

struct p4a_cl_kernel *p4a_search_current_kernel(const char *kernel)
{
  struct p4a_cl_kernel *kernel_in_the_list=p4a_kernels;
  while (kernel_in_the_list != NULL) {
    if (strcmp(kernel_in_the_list->name,kernel) == 0)
      return kernel_in_the_list;
    kernel_in_the_list = kernel_in_the_list->next;
  }
  return NULL;
}

#endif //!__cplusplus

/** Stop a timer on the accelerator and get float time in second

 @addtogroup P4A_opencl_time_measure
 */

double P4A_accel_copy_timer()
{
  cl_ulong start,end;
  double time;

#ifdef P4A_PROFILING
  if (p4a_event_copy) {
    clWaitForEvents(1, &p4a_event_copy);
    
    P4A_test_execution(clGetEventProfilingInfo(p4a_event_copy, 
					       CL_PROFILING_COMMAND_END, 
					       sizeof(cl_ulong), 
					       &end, 
					       NULL));
    P4A_test_execution(clGetEventProfilingInfo(p4a_event_copy, 
					       CL_PROFILING_COMMAND_START, 
					       sizeof(cl_ulong), 
					       &start, 
					       NULL));
    // execution_time in nanoseconds	      
    time = (float)(end - start);
    //printf("p4a_time_copy : %f\n",time*1.0e-9);
    // Return the time in second: 
    return time*1.0e-9;
  }
#endif
  return 0;
}

double P4A_accel_timer_stop_and_float_measure() 
{
  cl_ulong start,end;

#ifdef P4A_PROFILING
  if (p4a_event_execution) {
    clWaitForEvents(1, &p4a_event_execution);
    
    P4A_test_execution(clGetEventProfilingInfo(p4a_event_execution, 
					       CL_PROFILING_COMMAND_END, 
					       sizeof(cl_ulong), 
					       &end, 
					       NULL));
    P4A_test_execution(clGetEventProfilingInfo(p4a_event_execution, 
					       CL_PROFILING_COMMAND_START, 
					       sizeof(cl_ulong), 
					       &start, 
					       NULL));
    // execution_time in nanoseconds	      
    p4a_time_execution += (float)(end - start);
    // Return the time in second:
    return p4a_time_execution*1.0e-9;
  }
#endif
  return 0;
}

/** Allocate memory on the hardware accelerator in OpenCL mode.

    @param[out] address is the address of a variable that is updated by
    this function to contains the address of the allocated memory block

    @param[in] size is the size to allocate in bytes
*/
void P4A_accel_malloc(void **address, size_t size) {
  // The mem flag can be : 
  // CL_MEM_READ_ONLY, 
  // CL_MEM_WRITE_ONLY, 
  // CL_MEM_READ_WRITE, 
  // CL_MEM_USE_HOST_PTR
  *address=(void *)clCreateBuffer(p4a_context,
				  CL_MEM_READ_WRITE,
				  size,
				  NULL,
				  &p4a_global_error);
  P4A_test_execution_with_message("Memory allocation via clCreateBuffer");
}

/** Free memory on the hardware accelerator in OpenCL mode

    @param[in] address points to a previously allocated memory zone for
    the hardware accelerator
*/
void P4A_accel_free(void *address) 
{
  p4a_global_error=clReleaseMemObject((cl_mem)address);
  P4A_test_execution_with_message("clReleaseMemObject");
}

/** Copy a scalar from the hardware accelerator to the host

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
			 const void *accel_address) 
{
  p4a_global_error=clEnqueueReadBuffer(p4a_queue,
				       (cl_mem)accel_address,
				       CL_TRUE, // synchronous read
				       0,
				       element_size,
				       host_address,
				       0,
				       NULL,
				       &p4a_event_copy);
  p4a_time_copy += P4A_accel_copy_timer();
  P4A_test_execution_with_message("clEnqueueReadBuffer");
}

/** Copy a scalar from the host to the hardware accelerator

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
		       void *accel_address) 
{
  p4a_global_error=clEnqueueWriteBuffer(p4a_queue,
					(cl_mem)accel_address,
					CL_FALSE,// asynchronous write
					0,
					element_size,
					host_address,
					0,
					NULL,
					&p4a_event_copy);
  p4a_time_copy += P4A_accel_copy_timer();
  P4A_test_execution_with_message("clEnqueueWriteBuffer");
}

/** Function for copying memory from the hardware accelerator to a 1D array in
 the host.

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
			    const void *accel_address) 
{
  char * cdest = d1_offset*element_size + (char *)host_address;
  // CL_TRUE : synchronous read
  p4a_global_error=clEnqueueReadBuffer(p4a_queue,
				       (cl_mem)accel_address,
				       CL_TRUE,
				       0,
				       d1_block_size*element_size,
				       cdest,
				       0,
				       NULL,
				       &p4a_event_copy);
  p4a_time_copy += P4A_accel_copy_timer();
  P4A_test_execution_with_message("clEnqueueReadBuffer");
}


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
			  void *accel_address) 
{
  const char * csrc = d1_offset*element_size + (char *)host_address;
  p4a_global_error=clEnqueueWriteBuffer(p4a_queue,
					(cl_mem)accel_address,
					CL_FALSE, //CL_FALSE:asynchronous write
					0,
					d1_block_size*element_size,
					csrc,
					0,
					NULL,
					&p4a_event_copy);
  p4a_time_copy += P4A_accel_copy_timer();
  P4A_test_execution_with_message("clEnqueueWriteBuffer");
}

/** Function for copying memory from the hardware accelerator to a 2D array in
 the host.
 */
void P4A_copy_from_accel_2d(size_t element_size,
			    size_t d1_size, size_t d2_size,
			    size_t d1_block_size, size_t d2_block_size,
			    size_t d1_offset, size_t d2_offset,
			    void *host_address,
			    const void *accel_address) {

  if(d2_size==d2_block_size ) { // d2 is fully transfered ? We can optimize :-)
    // We transfer all in one shot !
    P4A_copy_from_accel_1d(element_size,
                           d2_size * d1_size,
                           d2_size * d1_block_size,
                           d1_offset*d2_size+d2_offset,
                           host_address,
                           accel_address);
  } else { // Transfer row by row !
    // Compute the adress of the begining of the first row to transfer
    char * host_row = (char*)host_address + d1_offset*d2_size*element_size;
    char * accel_row = (char*)accel_address;

    // Will transfert "d1_block_size" rows
    for(size_t i = 0; i < d1_block_size; i++) {
      P4A_copy_from_accel_1d(element_size,
                             d2_size,
                             d2_block_size,
                             d2_offset,
                             host_row,
                             accel_row);

      // Comput addresses for next row
      host_row += d2_size*element_size;
      accel_row += d2_block_size*element_size;
    }
  }
}

/** Function for copying a 2D memory zone from the host to a compact memory
 zone in the hardware accelerator.
 */
void P4A_copy_to_accel_2d(size_t element_size,
			  size_t d1_size, size_t d2_size,
			  size_t d1_block_size, size_t d2_block_size,
			  size_t d1_offset, size_t d2_offset,
			  const void *host_address,
			  void *accel_address) 
{
  if(d2_size==d2_block_size ) { // d2 is fully transfered ? We can optimize :-)
    // We transfer all in one shot !
    P4A_copy_to_accel_1d(element_size,
                         d2_size * d1_size,
                         d2_size * d1_block_size,
                         d1_offset*d2_size+d2_offset,
                         host_address,
                         accel_address);
  } else { // Transfer row by row !
    // Compute the adress of the begining of the first row to transfer
    char * host_row = (char*)host_address + d1_offset*d2_size*element_size;
    char * accel_row = (char*)accel_address;
    
    // Will transfert "d1_block_size" rows
    for(size_t i = 0; i < d1_block_size; i++) {
      P4A_copy_to_accel_1d(element_size,
                           d2_size,
                           d2_block_size,
                           d2_offset,
                           host_row,
                           accel_row);

      // Comput addresses for next row
      host_row += d2_size*element_size;
      accel_row += d2_block_size*element_size;
    }
  }

}

/** Function for copying memory from the hardware accelerator to a 3D array in
    the host.
*/
void P4A_copy_from_accel_3d(size_t element_size,
			    size_t d1_size, size_t d2_size, size_t d3_size,
			    size_t d1_block_size, 
			    size_t d2_block_size, size_t d3_block_size,
			    size_t d1_offset, 
			    size_t d2_offset, size_t d3_offset,
			    void *host_address,
			    const void *accel_address) {


  if(d3_size==d3_block_size ) { // d3 is fully transfered ? We can optimize :-)
    // We transfer all in one shot !
    P4A_copy_from_accel_2d(element_size,
                           d1_size,d3_size * d2_size,
                           d1_block_size, d3_size * d2_block_size,
                           d1_offset, d2_offset*d3_size+d3_offset,
                           host_address,
                           accel_address);
  } else { // Transfer row by row !
    // Compute the adress of the begining of the first row to transfer
    char * host_row = (char*)host_address + d1_offset*d2_size*d3_size*element_size;
    char * accel_row = (char*)accel_address;

    // Will transfert "d1_block_size" rows
    for(size_t i = 0; i < d1_block_size; i++) {
      P4A_copy_from_accel_2d(element_size,
                             d2_size,d3_size,
                             d2_block_size, d3_block_size,
                             d2_offset, d3_offset,
                             host_row,
                             accel_row);

      // Comput addresses for next row
      host_row += d3_size*element_size;
      accel_row += d3_block_size*element_size;
    }
  }
}


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
			  void *accel_address) 
{
  if(d3_size==d3_block_size ) { // d3 is fully transfered ? We can optimize :-)
    // We transfer all in one shot !
    P4A_copy_to_accel_2d(element_size,
                         d1_size,d3_size * d2_size,
                         d1_block_size, d3_size * d2_block_size,
                         d1_offset, d2_offset*d3_size+d3_offset,
                         host_address,
                         accel_address);
  } else { // Transfer row by row !
    // Compute the adress of the begining of the first row to transfer
    char * host_row = (char*)host_address + d1_offset*d2_size*d3_size*element_size;
    char * accel_row = (char*)accel_address;

    // Will transfert "d1_block_size" rows
    for(size_t i = 0; i < d1_block_size; i++) {
      P4A_copy_to_accel_2d(element_size,
                           d2_size,d3_size,
                           d2_block_size, d3_block_size,
                           d2_offset, d3_offset,
                           host_row,
                           accel_row);

      // Comput addresses for next row
      host_row += d3_size*element_size;
      accel_row += d3_block_size*element_size;
    }
  }
}

/** Function for copying memory from the hardware accelerator to a 4D array in
    the host.
*/
void P4A_copy_from_accel_4d(size_t element_size,
			    size_t d1_size, size_t d2_size, 
			    size_t d3_size, size_t d4_size,
			    size_t d1_block_size, size_t d2_block_size, 
			    size_t d3_block_size, size_t d4_block_size,
			    size_t d1_offset, size_t d2_offset, 
			    size_t d3_offset, size_t d4_offset,
			    void *host_address,
			    const void *accel_address) {

  if(d4_size==d4_block_size ) { // d4 is fully transfered ? We can optimize :-)
    // We transfer all in one shot !
    P4A_copy_from_accel_3d(element_size,
                         d1_size,d2_size,d3_size * d4_size,
                         d1_block_size, d2_block_size, d3_block_size * d4_size,
                         d1_offset, d2_offset, d3_offset*d4_size+d4_offset,
                         host_address,
                         accel_address);
  } else { // Transfer row by row !
    // Compute the adress of the begining of the first row to transfer
    char * host_row = (char*)host_address + d1_offset*d2_size*d3_size*d4_size*element_size;
    char * accel_row = (char*)accel_address;

    // Will transfert "d1_block_size" rows
    for(size_t i = 0; i < d1_block_size; i++) {
      P4A_copy_from_accel_3d(element_size,
                           d2_size,d3_size,d4_size,
                           d2_block_size, d3_block_size, d4_block_size,
                           d2_offset, d3_offset, d4_offset,
                           host_row,
                           accel_row);

      // Comput addresses for next row
      host_row += d4_size*element_size;
      accel_row += d4_block_size*element_size;
    }
  }
}

/** Function for copying a 4D memory zone from the host to a compact memory
    zone in the hardware accelerator.
*/
void P4A_copy_to_accel_4d(size_t element_size,
			  size_t d1_size, size_t d2_size, 
			  size_t d3_size, size_t d4_size,
			  size_t d1_block_size, size_t d2_block_size, 
			  size_t d3_block_size, size_t d4_block_size,
			  size_t d1_offset, size_t d2_offset, 
			  size_t d3_offset, size_t d4_offset,
			  const void *host_address,
			  void *accel_address) {
  if (d4_size == d4_block_size) { //d4 is fully transfered ? We can optimize :-)
    // We transfer all in one shot !
    P4A_copy_to_accel_3d(element_size,
                         d1_size,d2_size,d3_size * d4_size,
                         d1_block_size, d2_block_size, d3_block_size * d4_size,
                         d1_offset, d2_offset, d3_offset*d4_size+d4_offset,
                         host_address,
                         accel_address);
  } 
  else { // Transfer row by row !
    // Compute the adress of the begining of the first row to transfer
    char * host_row = (char*)host_address + d1_offset*d2_size*d3_size*d4_size*element_size;
    char * accel_row = (char*)accel_address;

    // Will transfert "d1_block_size" rows
    for(size_t i = 0; i < d1_block_size; i++) {
      P4A_copy_to_accel_3d(element_size,
                           d2_size,d3_size,d4_size,
                           d2_block_size, d3_block_size, d4_block_size,
                           d2_offset, d3_offset, d4_offset,
                           host_row,
                           accel_row);

      // Comput addresses for next row
      host_row += d4_size*element_size;
      accel_row += d4_block_size*element_size;
    }
  }
}

/** @author : Stéphanie Even

    In OpenCL, errorToString doesn't exists by default ...
    Macros have been picked from cl.h
 */
char * p4a_error_to_string(int error) 
{
  switch (error)
    {
    case CL_SUCCESS:
      return (char *)"P4A : Success";
    case CL_DEVICE_NOT_FOUND:
      return (char *)"P4A : Device Not Found";
    case CL_DEVICE_NOT_AVAILABLE:
      return (char *)"P4A : Device Not Available";
    case CL_COMPILER_NOT_AVAILABLE:
      return (char *)"P4A : Compiler Not Available";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:
      return (char *)"P4A : Mem Object Allocation Failure";
    case CL_OUT_OF_RESOURCES:
      return (char *)"P4A : Out Of Ressources";
    case CL_OUT_OF_HOST_MEMORY:
      return (char *)"P4A : Out Of Host Memory";
    case CL_PROFILING_INFO_NOT_AVAILABLE:
      return (char *)"P4A : Profiling Info Not Available";
    case CL_MEM_COPY_OVERLAP:
      return (char *)"P4A : Mem Copy Overlap";
    case CL_IMAGE_FORMAT_MISMATCH:
      return (char *)"P4A : Image Format Mismatch";
    case CL_IMAGE_FORMAT_NOT_SUPPORTED:
      return (char *)"P4A : Image Format Not Supported";
    case CL_BUILD_PROGRAM_FAILURE:
      return (char *)"P4A : Build Program Failure";
    case CL_MAP_FAILURE:
      return (char *)"P4A : Map Failure";
    case CL_INVALID_VALUE:
      return (char *)"P4A : Invalid Value";
    case CL_INVALID_DEVICE_TYPE:
      return (char *)"P4A : Invalid Device Type";
    case CL_INVALID_PLATFORM:
      return (char *)"P4A : Invalid Platform";
    case CL_INVALID_DEVICE:
      return (char *)"P4A : Invalid Device";
    case CL_INVALID_CONTEXT:
      return (char *)"P4A : Invalid Context";
    case CL_INVALID_QUEUE_PROPERTIES:
      return (char *)"P4A : Invalid Queue Properties";
    case CL_INVALID_COMMAND_QUEUE:
      return (char *)"P4A : Invalid Command Queue";
    case CL_INVALID_HOST_PTR:
      return (char *)"P4A : Invalid Host Ptr";
    case CL_INVALID_MEM_OBJECT:
      return (char *)"P4A : Invalid Mem Object";
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
      return (char *)"P4A : Invalid Image Format Descriptor";
    case CL_INVALID_IMAGE_SIZE:
      return (char *)"P4A : Invalid Image Size";
    case CL_INVALID_SAMPLER:
      return (char *)"P4A : Invalid Sampler";
    case CL_INVALID_BINARY:
      return (char *)"P4A : Invalid Binary";
    case CL_INVALID_BUILD_OPTIONS:
      return (char *)"P4A : Invalid Build Options";
    case CL_INVALID_PROGRAM:
      return (char *)"P4A : Invalid Program";
    case CL_INVALID_PROGRAM_EXECUTABLE:
      return (char *)"P4A : Invalid Program Executable";
    case CL_INVALID_KERNEL_NAME:
      return (char *)"P4A : Invalid Kernel Name";
    case CL_INVALID_KERNEL_DEFINITION:
      return (char *)"P4A : Invalid Kernel Definition";
    case CL_INVALID_KERNEL:
      return (char *)"P4A : Invalid Kernel";
    case CL_INVALID_ARG_INDEX:
      return (char *)"P4A : Invalid Arg Index";
    case CL_INVALID_ARG_VALUE:
      return (char *)"P4A : Invalid Arg Value";
    case CL_INVALID_ARG_SIZE:
      return (char *)"P4A : Invalid Arg Size";
    case CL_INVALID_KERNEL_ARGS:
      return (char *)"P4A : Invalid Kernel Args";
    case CL_INVALID_WORK_DIMENSION:
      return (char *)"P4A : Invalid Work Dimension";
    case CL_INVALID_WORK_GROUP_SIZE:
      return (char *)"P4A : Invalid Work Group Size";
    case CL_INVALID_WORK_ITEM_SIZE:
      return (char *)"P4A : Invalid Work Item Size";
    case CL_INVALID_GLOBAL_OFFSET:
      return (char *)"P4A : Invalid Global Offset";
    case CL_INVALID_EVENT_WAIT_LIST:
      return (char *)"P4A : Invalid Event Wait List";
    case CL_INVALID_EVENT:
      return (char *)"P4A : Invalid Event";
    case CL_INVALID_OPERATION:
      return (char *)"P4A : Invalid Operation";
    case CL_INVALID_GL_OBJECT:
      return (char *)"P4A : Invalid GL Object";
    case CL_INVALID_BUFFER_SIZE:
      return (char *)"P4A : Invalid Buffer Size";
    case CL_INVALID_MIP_LEVEL:
      return (char *)"P4A : Invalid Mip Level";
    case CL_INVALID_GLOBAL_WORK_SIZE:
      return (char *)"P4A : Invalid Global Work Size";
    default:
      break;
    }
} 

/** @author : Stéphanie Even

    To quit properly OpenCL.
 */
void p4a_clean(int exitCode)
{
  if(p4a_program)clReleaseProgram(p4a_program);
  if(p4a_kernel)clReleaseKernel(p4a_kernel);  
  if(p4a_queue)clReleaseCommandQueue(p4a_queue);
  if(p4a_context)clReleaseContext(p4a_context);
  exit(exitCode);
}

/** @author : Based on the CUDA version from Grégoire Péan. 
              Picked and adapted to OpenCL by Stéphanie Even

    Error display based on OpenCL error codes.

    Each OpenCL function returns an error information.
 */
void p4a_error(cl_int error, 
	       const char *currentFile, 
	       const int currentLine)
{
  if(CL_SUCCESS != error) {
    fprintf(stderr, "File %s - Line %i - The runtime error is %s\n",
	    currentFile,currentLine,P4A_error_to_string(error));
    p4a_clean(EXIT_FAILURE);
  }
#ifdef P4A_DEBUG
  else
    fprintf(stdout, "File %s - Line %i - %s\n",
	    currentFile,currentLine,P4A_error_to_string(error));
#endif
}

/** @author : Based on the CUDA version from Grégoire Péan. 
              Picked and adapted to OpenCL by Stéphanie Even

    Error display with message based on OpenCL error codes.
 */
void p4a_message(const char *message, 
		 const char *currentFile, 
		 const int currentLine)
{
  if(CL_SUCCESS != p4a_global_error){
    fprintf(stderr, "File %s - Line %i - Failed - %s : %s\n", currentFile, currentLine, message, P4A_error_to_string(p4a_global_error));
    p4a_clean(EXIT_FAILURE);
  }
#ifdef P4A_DEBUG
  else {
    fprintf(stdout, "File %s - Line %i - Success - %s\n", 
	    currentFile, currentLine, message);
  }
#endif 
}

/** @author : Stéphanie Even

    When launching the kernel
    - need to create the program from sources
    - select the kernel call within the program source

    Arguments are pushed from the ... list.
    Two solutions are proposed :
    - One, is actually commented : the parameters list (...)
      must contain the number of parameters, the sizeof(type)
      and the parameter as a reference &;
    - Second, the program source is parsed and the argument list is analysed.
      The result of the aprser is :
      - the number of arguments
      - the sizeof(argument type)

    @param kernel: Name of the kernel and the source MUST have the same
    name with .cl extension.
 */

void p4a_load_kernel(const char *kernel,...)
{
#ifdef __cplusplus
  std::string scpy(kernel);
  struct p4a_cl_kernel *k = p4a_kernels_map[scpy];
  
#else
  struct p4a_cl_kernel *k = p4a_search_current_kernel(kernel);
#endif
  // If not found ...
  if (!k) {
    P4A_log("The kernel %s is loaded for the first time\n",kernel);

#ifdef __cplusplus
    k = new p4a_cl_kernel(kernel);
    p4a_kernels_map[scpy] = k;
#else
    k = new_p4a_kernel(kernel);
#endif

    char* kernelFile =  k->file_name;
    P4A_log("Program and Kernel creation from %s\n",kernelFile);    
    size_t kernelLength;
    const char *comment = "// This kernel was generated for P4A\n";
    // Same design as the NVIDIA oclLoadProgSource
    char* cSourceCL = p4a_load_prog_source(kernelFile,
					   comment,
					   &kernelLength);
    if (cSourceCL == NULL)
      P4A_log("source du program null\n");
    else
      P4A_log("%s\n",cSourceCL);
    P4A_log("Kernel length = %lu\n",kernelLength);
    
    /*Create and compile the program : 1 for 1 kernel */
    p4a_program=clCreateProgramWithSource(p4a_context,1,
					  (const char **)&cSourceCL,
					  &kernelLength,
					  &p4a_global_error);
    P4A_test_execution_with_message("clCreateProgramWithSource");
    p4a_global_error=clBuildProgram(p4a_program,0,NULL,NULL,NULL,NULL);
    P4A_test_execution_with_message("clBuildProgram");
    p4a_kernel=clCreateKernel(p4a_program,kernel,&p4a_global_error);
    //current_kernel->kernel = p4a_kernel;
    k->kernel = p4a_kernel;
    P4A_test_execution_with_message("clCreateKernel");
    free(cSourceCL);
  }
  p4a_kernel = k->kernel;
}

/** @author : Stéphanie Even

    Load and store the content of the kernel file in a string.
    Replace the oclLoadProgSource function of NVIDIA.
 */
char *p4a_load_prog_source(char *cl_kernel_file,const char *head,size_t *length)
{
  // Initialize the size and memory space
  struct stat buf;
  stat(cl_kernel_file,&buf);
  size_t size = buf.st_size;
  size_t len = strlen(head);
  char *source = (char *)malloc(len+size+1);
  strncpy(source,head,len);

  // A string pointer referencing to the position after the head
  // where the storage of the file content must begin
  char *p = source+len;

  // Open the file
  int in = open(cl_kernel_file,O_RDONLY);
  if (!in) 
    P4A_log_and_exit(EXIT_FAILURE,"Bad kernel source reference : %s\n",cl_kernel_file);
  
  // Read the file content
  int n=0;
  if ((n = read(in,(void *)p,size)) != (int)size) 
    P4A_log_and_exit(EXIT_FAILURE,"Read was not completed : %d / %lu octets\n",n,size);
  
  // Final string marker
  source[len+n]='\0';
  close(in);
  *length = size+len;
  return source;
}

#endif // P4A_ACCEL_OPENCL
