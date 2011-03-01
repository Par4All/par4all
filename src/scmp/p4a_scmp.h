#ifndef P4A_SCMP_H
#define P4A_SCMP_H

#include <stddef.h>

extern void P4A_scmp_reset();
extern void P4A_scmp_malloc(void **dest,  size_t n, 
                     size_t dest_id, int do_it, int producer_pd);

extern void P4A_scmp_dealloc(void *dest, 
                      size_t dest_id, int do_it, int producer_p);

extern void P4A_copy_from_accel_1d_server(size_t element_size,
			    size_t d1_size,
			    size_t d1_block_size,
			    size_t d1_offset,
			    void *host_address,
			    const void *accel_address,
			    unsigned int accel_address_id,
                            int do_it);

extern void P4A_copy_to_accel_1d_server(size_t element_size,
			  size_t d1_size,
			  size_t d1_block_size,
			  size_t d1_offset,
			  const void *host_address,
			  void *accel_address,
			  unsigned int accel_address_id,
                          int do_it);
extern void P4A_copy_from_accel_1d(size_t element_size,
			    size_t d1_size,
			    size_t d1_block_size,
			    size_t d1_offset,
			    void *host_address,
			    const void *accel_address,
			    unsigned int accel_address_id,
                            int do_it);

extern void P4A_copy_to_accel_1d(size_t element_size,
			  size_t d1_size,
			  size_t d1_block_size,
			  size_t d1_offset,
			  const void *host_address,
			  void *accel_address,
			  unsigned int accel_address_id,
                          int do_it);
#endif
