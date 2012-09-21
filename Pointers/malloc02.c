/* Check type sensitivity. */

#include <malloc.h>

int main(int argc, char *argv[])
{
  int * pi;
  float * px;
  char * pc = argv[0];

  pi = (int *) malloc(sizeof(int));
  px = (float *) malloc(sizeof(float));

  return (int) *pc;
}
