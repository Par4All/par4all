/* Short version of io_intrisincs.c to debug the mysterious import of comments
   from the header file into main "fmt2=..." statement */

#include <stdlib.h>
#include <stdarg.h>
#include <stdio.h>

int main(char *fmt1, ...)
{
  char *buf2 = malloc(200 * sizeof(char));
  char * fmt2;


  // formatted IO functions
  fmt2 = "%s%d";
}
