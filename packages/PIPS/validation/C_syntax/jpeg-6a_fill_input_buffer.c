
#include <stdlib.h>
#include <stdio.h>
typedef unsigned char JOCTET;

struct jpeg_source_mgr {
  const JOCTET * next_input_byte; /* => next byte to read from buffer */
  size_t bytes_in_buffer;	/* # of bytes remaining in buffer */

};


struct jpeg_decompress_struct {
  struct jpeg_source_mgr * src;

};


typedef struct {
  struct jpeg_source_mgr pub;	/* public fields */

  FILE * infile;		/* source stream */
  JOCTET * buffer;		/* start of buffer */
  int start_of_file;	/* have we gotten any data yet? */
} my_source_mgr;

typedef my_source_mgr * my_src_ptr;


typedef struct jpeg_decompress_struct * j_decompress_ptr;

int
fill_input_buffer (j_decompress_ptr cinfo)
{
  my_src_ptr src = (my_src_ptr) cinfo->src;
  size_t nbytes;

  nbytes = (size_t) fread((void *) src->buffer, (size_t) 1, (size_t) 4096, src->infile);

  return 1;
}
