#include <stdlib.h>

/** @defgroup p4a_accel_stubs Equivalent stubs of Par4All runtime to have
    PIPS analyzes happy

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

  @{
 */


/** Stub for copying memory from the host to the hardware accelerator.

    Since it is a stub so that PIPS can understand it, use simple
    implementation with standard memory copy operations

    Do not change the place of the pointers in the API. The host address
    is always in the first position...

    @param[in] host_address is the address of a source zone in the host memory

    @param[out] accel_address is the address of a destination zone in the
    accelerator memory

    @param[in] size is the size in bytes of the memory zone to copy
*/
void * P4A_copy_to_accel(const void * host_address,
			 void * accel_address,
			 size_t size) {
  size_t i;

  for(i = 0 ; i < size; i++)
    ((char*)accel_address)[i] = ((const char*)host_address)[i];
  return accel_address;
}


/** Stub for copying memory from the hardware accelerator to the host.

    Do not change the place of the pointers in the API. The host address
    is always in the first position...

    @param[out] host_address is the address of a destination zone in the
    host memory

    @param[in] accel_address is the address of a source zone in the
    accelerator memory

    @param[in] size is the size in bytes of the memory zone to copy
*/
void * P4A_copy_from_accel(void * host_address,
			   const void * accel_address,
			   size_t size) {
  size_t i;

  for(i = 0; i < size; i++)
    ((char*)host_address)[i] = ((const char*)accel_address)[i];
  return host_address;
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

/** @} */

/** @} */
