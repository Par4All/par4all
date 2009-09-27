#include <stdlib.h>

/* Equivalent stubs of Par4All runtime to have PIPS analyze happy */

void * P4A_COPY_TO_ACCEL(const void * host_address,
			 void * accel_address,
			 size_t n) {
  size_t i;

  for(i = 0 ; i < n; i++)
    ((char*)accel_address)[i] = ((const char*)host_address)[i];
  return accel_address;
}


void * P4A_COPY_FROM_ACCEL(void * host_address,
			   const void * accel_address,
			   size_t n) {
  size_t i;

  for(i=0;i<n;i++)
    ((char*)host_address)[i] = ((const char*)accel_address)[i];
  return host_address;
}


void P4A_ACCEL_MALLOC(void **dest,  size_t n) {
  *dest = malloc(n);
}


void P4A_ACCEL_FREE(void *dest) {
  free(dest);
}
