#include <sesam_com.h>
#include "p4a_scmp.h"
#include "scmp_buffers.h" // defines NB_BUFFERS

size_t current_buffer_size[NB_BUFFERS];


#define P4A_sesam_debug(format, args...)\
   sesam_printf(format, ##args)

void P4A_scmp_reset()
{
  int i;
  for (i = 0; i <NB_BUFFERS; i ++)
    {
      current_buffer_size[i] = 0;
    }
}

static void P4A_sesam_wait_all_pages(unsigned int id, int size, int RW)
{
  int i;
  for (i = 0; i < size / sesam_get_page_size() + 1; i++)
    sesam_wait_page(id, i, RW, 0);
}

static void P4A_sesam_send_all_pages(unsigned int id, int size, int RW)
{
  int i;
  for (i = 0; i < size/sesam_get_page_size()+1; i++)
    sesam_send_page(id, i, RW);
}

void P4A_scmp_malloc(void **dest,  size_t n, 
                     unsigned int dest_id, int do_it, int producer_p)
{
   //P4A_sesam_debug("malloc, id = %d, do_it = %d, producer_p = %d\n", dest_id, do_it, producer_p);
  if(do_it)
  {
    if (producer_p)
    {
      size_t current_size = current_buffer_size[dest_id];
      //P4A_sesam_debug("current_size = %d\n", current_size);
      if (current_size != 0)
      {
	//P4A_sesam_debug("waiting\n");
	P4A_sesam_wait_all_pages(dest_id, current_size, O_WRITE);
      }
      if (current_size !=0 && current_size != n)
      {
	//P4A_sesam_debug("freeing\n");
        sesam_free_data(dest_id);
        current_size = 0;
      }
      if (current_size == 0)
      {
 	size_t i;
	 //P4A_sesam_debug("allocating\n");
        sesam_reserve_data(n/sesam_get_page_size()+1);
        sesam_data_assignation(dest_id, n/sesam_get_page_size()+1, 1);
	P4A_sesam_wait_all_pages(dest_id, n, O_WRITE);
     }
    }
    current_buffer_size[dest_id] = n;
    *dest = sesam_map_data(dest_id);
    //P4A_sesam_debug("mapping to %p\n", *dest);
  }
}

void P4A_scmp_dealloc(void *dest, 
                      unsigned int dest_id, int do_it, int producer_p)
{
  //P4A_sesam_debug("dealloc, id = %d, do_it = %d, producer_p = %d\n", dest_id, do_it, producer_p);
  if (do_it)
  {
    if (!producer_p)
    {
      size_t current_size = current_buffer_size[dest_id];
      //P4A_sesam_debug("sending\n");
      P4A_sesam_send_all_pages(dest_id, current_size, O_WRITE);
    }
     //P4A_sesam_debug("unmapping %p\n", dest);
    sesam_unmap_data(dest);
  }
}

void P4A_copy_from_accel_1d_server(size_t element_size,
			    size_t d1_size,
			    size_t d1_block_size,
			    size_t d1_offset,
			    void *host_address,
			    const void *accel_address,
			    unsigned int accel_address_id,
                            int do_it)
{
  int i;
   //P4A_sesam_debug("P4A_copy_from_accel_1d_server, data_id = %d, do_it = %d\n", accel_address_id, do_it);
  if (do_it)
  {
     //P4A_sesam_debug("waiting for read access\n");
    P4A_sesam_wait_all_pages(accel_address_id, d1_block_size*element_size, O_READ);

    char * cdest = d1_offset*element_size + (char *)host_address;
    const char * csrc = accel_address;
    for(i = 0; i < d1_block_size*element_size; i++)
      cdest[i] = csrc[i];
  }
   //P4A_sesam_debug("P4A_copy_from_accel_1d_server, end\n");
}

void P4A_copy_to_accel_1d_server(size_t element_size,
			  size_t d1_size,
			  size_t d1_block_size,
			  size_t d1_offset,
			  const void *host_address,
			  void *accel_address,
                          unsigned int accel_address_id,
			  int do_it)
{
  size_t i;
   //P4A_sesam_debug("P4A_copy_to_accel_1d_server, data_id = %d, do_it = %d\n", accel_address_id, do_it);
 if (do_it)
  {
    char * cdest = accel_address;
    const char * csrc = d1_offset*element_size + (char *)host_address;
    for(i = 0; i < d1_block_size*element_size; i++)
      cdest[i] = csrc[i];
     //P4A_sesam_debug("sending\n");
    P4A_sesam_send_all_pages(accel_address_id, d1_block_size*element_size, O_READ);
  }
   //P4A_sesam_debug("P4A_copy_to_accel_1d_server, end\n");

}

void P4A_copy_from_accel_1d(size_t element_size,
			    size_t d1_size,
			    size_t d1_block_size,
			    size_t d1_offset,
			    void *host_address,
			    const void *accel_address,
			    unsigned int accel_address_id,
                            int do_it)
{
   //P4A_sesam_debug("P4A_copy_from_accel_1d, data_id = %d, do_it = %d\n", accel_address_id, do_it);
  if (do_it)
    {
     //P4A_sesam_debug("sending\n");
      P4A_sesam_send_all_pages(accel_address_id, element_size*d1_block_size, O_READ);
    }
   //P4A_sesam_debug("P4A_copy_from_accel_1d, end\n");
}

void P4A_copy_to_accel_1d(size_t element_size,
			  size_t d1_size,
			  size_t d1_block_size,
			  size_t d1_offset,
			  const void *host_address,
			  void *accel_address,
			  unsigned int accel_address_id,
                          int do_it)
{
   //P4A_sesam_debug("P4A_copy_to_accel_1d, data_id = %d, do_it = %d\n", accel_address_id, do_it);
  if (do_it)
  {
     //P4A_sesam_debug("waiting\n");
    P4A_sesam_wait_all_pages(accel_address_id, element_size* d1_block_size, O_READ);
  }
   //P4A_sesam_debug("P4A_copy_to_accel_1d, end\n");
}


/** Stub for copying a 2D memory zone from the host to a compact memory
    zone in the hardware accelerator.
*/
void P4A_copy_to_accel_2d_server(size_t element_size,
			  size_t d1_size, size_t d2_size,
			  size_t d1_block_size, size_t d2_block_size,
			  size_t d1_offset,   size_t d2_offset,
			  const void *host_address,
			  void *accel_address,
			  unsigned int accel_address_id,
                          int do_it) {
   //P4A_sesam_debug("P4A_copy_to_accel_2d_server, data_id = %d, do_it = %d\n", accel_address_id, do_it);
  if (do_it)
    {
      size_t i, j;
      char * cdest = (char *)accel_address;
      const char * csrc = d2_offset*element_size + (char *)host_address;
      for(i = 0; i < d1_block_size; i++)
	for(j = 0; j < d2_block_size*element_size; j++)
	  cdest[i*element_size*d2_block_size + j] =
	    csrc[(i + d1_offset)*element_size*d2_size + j];
      //P4A_sesam_debug("sending\n");
      P4A_sesam_send_all_pages(accel_address_id,
			       d1_block_size*d2_block_size*element_size,
			       O_READ);
    }
  //P4A_sesam_debug("P4A_copy_to_accel_2d_server, end\n");
}


/** Stub for copying memory from the hardware accelerator to a 2D array in
    the host.
*/
void P4A_copy_from_accel_2d_server(size_t element_size,
				   size_t d1_size, size_t d2_size,
				   size_t d1_block_size, size_t d2_block_size,
				   size_t d1_offset, size_t d2_offset,
				   void *host_address,
				   const void *accel_address,
				   unsigned int accel_address_id,
				   int do_it) {
  //P4A_sesam_debug("P4A_copy_from_accel_2d_server, data_id = %d, do_it = %d\n", accel_address_id, do_it);
  if (do_it)
    {
      //P4A_sesam_debug("waiting for read access\n");
      P4A_sesam_wait_all_pages(accel_address_id,
			       d1_block_size*d2_block_size*element_size,
			       O_READ);

      size_t i, j;
      char * cdest = d2_offset*element_size + (char*)host_address;
      char * csrc = (char*)accel_address;
      for(i = 0; i < d1_block_size; i++)
	for(j = 0; j < d2_block_size*element_size; j++)
	  cdest[(i + d1_offset)*element_size*d2_size + j] =
	    csrc[i*element_size*d2_block_size + j];
    }
  //P4A_sesam_debug("P4A_copy_from_accel_2d_server, end\n");
 }

/** Stub for copying a 2D memory zone from the host to a compact memory
    zone in the hardware accelerator.
*/
void P4A_copy_to_accel_2d(size_t element_size,
			  size_t d1_size, size_t d2_size,
			  size_t d1_block_size, size_t d2_block_size,
			  size_t d1_offset,   size_t d2_offset,
			  const void *host_address,
			  void *accel_address,
			  unsigned int accel_address_id,
                          int do_it) {
   //P4A_sesam_debug("P4A_copy_to_accel_2d, data_id = %d, do_it = %d\n", accel_address_id, do_it);
  if (do_it)
  {
     //P4A_sesam_debug("waiting\n");
    P4A_sesam_wait_all_pages(accel_address_id,
			     element_size*d1_block_size*d2_block_size,
			     O_READ);
  }
   //P4A_sesam_debug("P4A_copy_to_accel_2d, end\n");
}


/** Stub for copying memory from the hardware accelerator to a 2D array in
    the host.
*/
void P4A_copy_from_accel_2d(size_t element_size,
			    size_t d1_size, size_t d2_size,
			    size_t d1_block_size, size_t d2_block_size,
			    size_t d1_offset, size_t d2_offset,
			    void *host_address,
			    const void *accel_address,
			    unsigned int accel_address_id,
			    int do_it) {
   //P4A_sesam_debug("P4A_copy_from_accel_2d, data_id = %d, do_it = %d\n", accel_address_id, do_it);
  if (do_it)
    {
     //P4A_sesam_debug("sending\n");
      P4A_sesam_send_all_pages(accel_address_id,
			       element_size*d1_block_size*d2_block_size,
			       O_READ);
    }
   //P4A_sesam_debug("P4A_copy_from_accel_2d, end\n");
}
