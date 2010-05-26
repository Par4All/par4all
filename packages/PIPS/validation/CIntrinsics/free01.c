/* free example :This program has no output. Just demonstrates some ways to allocate and free dynamic memory using the cstdlib functions.*/

#include <stdio.h>
#include <stdlib.h>

int main ()
{
  int * buffer1, * buffer2, * buffer3;
  buffer1 = (int*) malloc (100*sizeof(int));
  buffer2 = (int*) calloc (100,sizeof(int));
  buffer3 = (int*) realloc (buffer2,500*sizeof(int));
  free (buffer1);
  free (buffer3);
  return 0;
}
