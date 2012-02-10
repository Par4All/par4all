#include <stdlib.h>
#include <p4a_stacksize_test.h>

/**
 * Flag that trigger timing of kernel execution
 */
int p4a_timing = 0;

/**
 *  This is used by P4A_TIMING routine to compute elapsed time for a kernel
 *  execution (in ms)
 */
float p4a_timing_elapsedTime = -1;
struct timeval p4a_time_begin, p4a_time_end;

/**
 * The main debug level for Par4All Runtime
 */
int p4a_debug_level = 0;

/* Flag that trigger the Par4All runtime initialization */
int p4a_runtime_initialized = 0;

void p4a_main_init() {
  if(p4a_runtime_initialized) {
    P4A_dump_message("Runtime is initialized more than one time, it probably not intended ! Please report.\n");
  }

  p4a_runtime_initialized = 1;
  checkStackSize();

  /* Debug level stuff */
  char *env_p4a_debug= getenv ("P4A_DEBUG");
  if(env_p4a_debug) {
    P4A_dump_message("P4A_DEBUG environment variable is set : '%s'\n",env_p4a_debug);
    errno = 0;
    int debug_level = strtol(env_p4a_debug, NULL,10);
    if(errno==0) {
      P4A_dump_message("Setting debug level to %d\n",debug_level);
      p4a_debug_level= debug_level;
    } else {
      fprintf(stderr,"Invalid value for P4A_DEBUG: %s\n",env_p4a_debug);
    }
  }

  /* Timing stuff */
  char *env_p4a_timing= getenv ("P4A_TIMING");
  if(env_p4a_timing) {
    P4A_dump_message("P4A_TIMING environment variable is set : '%s'\n",env_p4a_timing);
    errno = 0;
    int timing = strtol(env_p4a_timing, NULL,10);
    if(errno==0) {
      P4A_dump_message("Enable timing (%d)\n",timing);
      p4a_timing= timing;
    } else {
      fprintf(stderr,"Invalid value for P4A_TIMING: %s\n",env_p4a_timing);
    }
  }
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
    const void *accel_address) {
  const char * cdest = d1_offset*element_size + (char *)host_address;
  P4A_copy_from_accel(d1_block_size * element_size, (void *)cdest, accel_address);
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
    void *accel_address) {
  const char * csrc = d1_offset*element_size + (char *)host_address;
  P4A_copy_to_accel(d1_block_size * element_size, csrc, accel_address);
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
    void *accel_address) {
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
          size_t d1_block_size, size_t d2_block_size, size_t d3_block_size,
          size_t d1_offset, size_t d2_offset, size_t d3_offset,
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
        size_t d1_block_size, size_t d2_block_size, size_t d3_block_size,
        size_t d1_offset,   size_t d2_offset, size_t d3_offset,
        const void *host_address,
        void *accel_address) {

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
          size_t d1_size, size_t d2_size, size_t d3_size, size_t d4_size,
          size_t d1_block_size, size_t d2_block_size, size_t d3_block_size, size_t d4_block_size,
          size_t d1_offset, size_t d2_offset, size_t d3_offset, size_t d4_offset,
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
          size_t d1_size, size_t d2_size, size_t d3_size, size_t d4_size,
          size_t d1_block_size, size_t d2_block_size, size_t d3_block_size, size_t d4_block_size,
          size_t d1_offset, size_t d2_offset, size_t d3_offset, size_t d4_offset,
          const void *host_address,
          void *accel_address) {
  if(d4_size==d4_block_size ) { // d4 is fully transfered ? We can optimize :-)
    // We transfer all in one shot !
    P4A_copy_to_accel_3d(element_size,
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
