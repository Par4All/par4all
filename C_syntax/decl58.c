// Extended version of decl57, such as provided by Beatrice Creusillet
// Mismanagement of const attribute

#include <stdlib.h>
#include <stdio.h>

/* main function added to avoid output dependences on stdlib.h and
   stdio.h */

int main()
{
  typedef unsigned char JOCTET;

  struct jpeg_source_mgr {
    const JOCTET * next_input_byte;
    size_t bytes_in_buffer;

  };

  typedef struct {
    struct jpeg_source_mgr pub;

    FILE * infile;
    JOCTET * buffer;
    int start_of_file;
  } my_source_mgr;
}
