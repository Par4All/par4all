#include <stdlib.h>

/** @defgroup p4a_accel_stubs Equivalent stubs of Par4All runtime to have
    PIPS analyzis happy

    The main idea is that PIPS is interprocedural and needs to know
    everything about the program to be analysed. Since the generated
    function calls are in the PIPS output, they need to be analyzed and so
    need to be in the PIPS input.

    Since CUDA runtime is not C or Fortran, it cannot be analyzed by
    PIPS. So, in the entry, we give to PIPS equivalent functions to eat
    (these stubs functions). Of course it is useless to have a functional
    emulation of each accelerator element processors :-), so just have a
    functional equivalent with the same behaviour on the memory (store
    effects in the PIPS jargon).

  @{
 */


/** @defgroup p4a_accel_C_stubs Equivalent stubs in C of Par4All runtime to have
    PIPS analyzes happy

    TODO: Well, all these stubs should be sourced from the OpenMP
    implementation from p4a_accel.c instead of being pasted here

    @{
*/


/** Stub for P4A_accel_kernel_wrapper #define in p4a_accel, which has to be
 * parsed as a typedef in pips
 */
typedef void P4A_accel_kernel_wrapper;
typedef void P4A_accel_kernel;


/** Stub for copying a scalar from the hardware accelerator memory to
    the host.

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
			 const void *accel_address) {
  size_t i;
  char * cdest = host_address;
  const char * csrc = accel_address;
  for(i = 0; i < element_size; i++)
    cdest[i] = csrc[i];
}


/** Stub for copying a scalar from the host to a memory zone in the
    hardware accelerator.
*/
void P4A_copy_to_accel(size_t element_size,
		       const void *host_address,
		       void *accel_address) {
  size_t i;
  char * cdest = accel_address;
  const char * csrc = host_address;
  for(i = 0; i < element_size; i++)
    cdest[i] = csrc[i];
}

/** Stub for copying memory from the host to the hardware accelerator.

    Since it is a stub so that PIPS can understand it, use simple
    implementation with standard memory copy operations

    Do not change the place of the pointers in the API. The host address
    is always in the first position...

    This function could be quite simpler but is designed by symmetry with
    other functions.

    @param[in] element_size is the size of one element of the array in
    byte

    @param[in] d1_size is the number of elements in the array. It is not
    used but here for symmetry with functions of higher dimensionality

    @param[in] d1_size is the number of elements in the array. It is not
    used but here for symmetry with functions of higher dimensionality

    @param[in] d1_block_size is the number of element to transfer

    @param[in] d1_offset is element order to start the transfer to

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
			    const void *accel_address) {
  size_t i;
  char * cdest = d1_offset*element_size + (char *)host_address;
  const char * csrc = accel_address;
  for(i = 0; i < d1_block_size*element_size; i++)
    cdest[i] = csrc[i];
}


/** Stub for copying a 1D memory zone from the hardware accelerator to the
    host.

    Do not change the place of the pointers in the API. The host address
    is always in the first position...

    This function could be quite simpler but is designed by symmetry with
    other functions.

    @param[in] element_size is the size of one element of the array in
    byte

    @param[in] d1_size is the number of elements in the array. It is not
    used but here for symmetry with functions of higher dimensionality

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
			  void *accel_address) {
  size_t i;
  char * cdest = accel_address;
  const char * csrc = d1_offset*element_size + (char *)host_address;
  for(i = 0; i < d1_block_size*element_size; i++)
    cdest[i] = csrc[i];
}


/** Stub for copying a 2D memory zone from the host to a compact memory
    zone in the hardware accelerator.
*/
void P4A_copy_to_accel_2d(size_t element_size,
			  size_t d1_size, size_t d2_size,
			  size_t d1_block_size, size_t d2_block_size,
			  size_t d1_offset,   size_t d2_offset,
			  const void *host_address,
			  void *accel_address) {
  size_t i, j;
  char * cdest = (char *)accel_address;
  const char * csrc = d2_offset*element_size + (char *)host_address;
  for(i = 0; i < d1_block_size; i++)
    for(j = 0; j < d2_block_size*element_size; j++)
      cdest[i*element_size*d2_block_size + j] =
        csrc[(i + d1_offset)*element_size*d2_size + j];
}


/** Stub for copying memory from the hardware accelerator to a 2D array in
    the host.
*/
void P4A_copy_from_accel_2d(size_t element_size,
			    size_t d1_size, size_t d2_size,
			    size_t d1_block_size, size_t d2_block_size,
			    size_t d1_offset, size_t d2_offset,
			    void *host_address,
			    const void *accel_address) {
  size_t i, j;
  char * cdest = d2_offset*element_size + (char*)host_address;
  char * csrc = (char*)accel_address;
  for(i = 0; i < d1_block_size; i++)
    for(j = 0; j < d2_block_size*element_size; j++)
      cdest[(i + d1_offset)*element_size*d2_size + j] =
        csrc[i*element_size*d2_block_size + j];
}


/** Stub for copying memory from the hardware accelerator to a 3D array in
    the host.
*/
void P4A_copy_from_accel_3d(size_t element_size,
                            size_t d1_size, size_t d2_size, size_t d3_size,
                            size_t d1_block_size, size_t d2_block_size, size_t d3_block_size,
                            size_t d1_offset, size_t d2_offset, size_t d3_offset,
                            void *host_address,
                            const void *accel_address) {
  size_t i, j, k;
  char * cdest = d3_offset*element_size + (char*)host_address;
  const char * csrc = (char*)accel_address;
  for(i = 0; i < d1_block_size; i++)
    for(j = 0; j < d2_block_size; j++)
      for(k = 0; k < d3_block_size*element_size; k++)
        cdest[((i + d1_offset)*d2_block_size + j + d2_offset)*element_size*d3_size + k] =
            csrc[(i*d2_block_size + j)*d3_block_size*element_size + k];
}


/** Stub for copying a 3D memory zone from the host to a compact memory
    zone in the hardware accelerator.
*/
void P4A_copy_to_accel_3d(size_t element_size,
                          size_t d1_size, size_t d2_size, size_t d3_size,
                          size_t d1_block_size, size_t d2_block_size, size_t d3_block_size,
                          size_t d1_offset,   size_t d2_offset, size_t d3_offset,
                          const void *host_address,
                          void *accel_address) {
  size_t i, j, k;
  char * cdest = (char *)accel_address;
  const char * csrc = d3_offset*element_size + (char *)host_address;
  for(i = 0; i < d1_block_size; i++)
    for(j = 0; j < d2_block_size; j++)
      for(k = 0; k < d3_block_size*element_size; k++)
        cdest[(i*d2_block_size + j)*d3_block_size*element_size + k] =
            csrc[((i + d1_offset)*d2_block_size + j + d2_offset)*element_size*d3_size + k];
}


/** Stub for copying memory from the hardware accelerator to a 4D array in
    the host.
*/
void P4A_copy_from_accel_4d(size_t element_size,
          size_t d1_size, size_t d2_size, size_t d3_size, size_t d4_size,
          size_t d1_block_size, size_t d2_block_size, size_t d3_block_size, size_t d4_block_size,
          size_t d1_offset, size_t d2_offset, size_t d3_offset, size_t d4_offset,
          void *host_address,
          const void *accel_address) {
  size_t i, j, k,l;
  char * cdest = (char*)host_address;
  const char * csrc = (char*)accel_address;
  for(i = 0; i < d1_block_size; i++) {
    for(j = 0; j < d2_block_size; j++) {
      for(k = 0; k < d3_block_size; k++) {
        for(l = 0; l < d4_block_size; l++) {
          int h_index = (i+d1_offset)*d2_size*d3_size*d4_size
              + (j + d2_offset )*d3_size*d4_size
              + (k + d3_offset )*d4_size
              + (l + d4_offset);
          int a_index = i*d2_block_size*d3_block_size*d4_block_size
              + j*d3_block_size*d4_block_size
              + k*d4_block_size
              + l;
          cdest[h_index] = csrc[a_index];
        }
      }
    }
  }
}


/** Stub for copying a 4D memory zone from the host to a compact memory
    zone in the hardware accelerator.
*/
void P4A_copy_to_accel_4d(size_t element_size,
        size_t d1_size, size_t d2_size, size_t d3_size, size_t d4_size,
        size_t d1_block_size, size_t d2_block_size, size_t d3_block_size, size_t d4_block_size,
        size_t d1_offset, size_t d2_offset, size_t d3_offset, size_t d4_offset,
        const void *host_address,
        void *accel_address) {
  size_t i, j, k,l;
  char * cdest = (char *)accel_address;
  const char * csrc = (char *)host_address;
  for(i = 0; i < d1_block_size; i++) {
    for(j = 0; j < d2_block_size; j++) {
      for(k = 0; k < d3_block_size; k++) {
        for(l = 0; l < d4_block_size; l++) {
          int h_index = (i+d1_offset)*d2_size*d3_size*d4_size
              + (j + d2_offset )*d3_size*d4_size
              + (k + d3_offset )*d4_size
              + (l + d4_offset);
          int a_index = i*d2_block_size*d3_block_size*d4_block_size
              + j*d3_block_size*d4_block_size
              + k*d4_block_size
              + l;
          cdest[a_index] = csrc[h_index];
        }
      }
    }
  }
}

/** Stub for allocating memory on the hardware accelerator.

    @param[out] address is the address of a variable that is updated by
    this macro to contains the address of the allocated memory block

    @param[in] size is the size to allocate in bytes
*/
void P4A_accel_malloc(void **address,  size_t size) {
  *address = malloc(size);
}


/** Stub for freeing memory on the hardware accelerator.

    @param[in] address is the address of a previously allocated memory
    zone on the hardware accelerator
*/
void P4A_accel_free(void *address) {
  free(address);
}


/** Stubs for P4A runtime */
void P4A_runtime_copy_to_accel(void *host_ptr, size_t size /* in bytes */) {
  P4A_copy_to_accel(size,host_ptr,NULL);
}
void P4A_runtime_copy_from_accel(void *host_ptr, size_t size /* in bytes */) {
  P4A_copy_from_accel(size,host_ptr,NULL);
}
void *P4A_runtime_host_ptr_to_accel_ptr(const void *host_ptr, size_t size) {
  void *accel_ptr;
  malloc(size);
  P4A_accel_malloc(&accel_ptr, size);
  return accel_ptr;
}




#ifdef P4A_RUNTIME_FFTW
#include <fftw3.h>
/*
typedef struct fftwf_plan_s {
  int x,y,z;
  fftwf_complex *in;
  fftwf_complex *out;
  int direction;
  int params;
  size_t datasize;
} fftwf_plan;
*/
void fftwf_free(void *p) {
  free(p);
}

void fftwf_destroy_plan( fftwf_plan p) {
  free(p);
}

static int fftwf_execute_x,fftwf_execute_y,fftwf_execute_z;
void fftwf_execute(fftwf_plan plan) {
  int i,j,k;
  for(i=0; i<fftwf_execute_x;i++) {
    for(j=0; j<fftwf_execute_y;j++) {
      for(k=0; k<fftwf_execute_z;k++) {
//        plan->out[(i*fftwf_execute_y*fftwf_execute_z)+j*fftwf_execute_z+k][0] = plan->in[(i*fftwf_execute_y*fftwf_execute_z)+j*fftwf_execute_z+k][0];
//        plan->out[(i*fftwf_execute_y*fftwf_execute_z)+j*fftwf_execute_z+k][1] = plan->in[(i*fftwf_execute_y*fftwf_execute_z)+j*fftwf_execute_z+k][1];
      }
    }
  }
}

fftwf_plan fftwf_plan_dft_3d(int nx, int ny, int nz, fftwf_complex *in, fftwf_complex *out, int direction, unsigned int params) {
  fftwf_plan plan;
  return plan;
}

void* fftwf_malloc(size_t size) {
 return malloc(size);
}



#endif // P4A_RUNTIME_FFTW





/** @} */

/** @} */
