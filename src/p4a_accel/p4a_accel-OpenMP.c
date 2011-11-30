#ifdef P4A_ACCEL_OPENMP

void p4a_init_openmp_accel() {
  p4a_main_init();
}

/**
 * global structure used for timing purpose
 */





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
void P4A_accel_malloc(void **address, size_t size) {
  *address = malloc(size);
}


/** Free memory on the hardware accelerator in OpenMP emulation.

    It is on the host too

    @param[in] address points to a previously allocated memory zone for
    the hardware accelerator
*/
void P4A_accel_free(void *address) {
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
    const void *accel_address) {
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
    void *accel_address) {
  /* We can use memcpy() since we are sure there is no overlap */
  memcpy(accel_address, host_address, element_size);
}

/* @} */
#endif // P4A_ACCEL_OPENMP
