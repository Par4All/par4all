/* Check type sensitivity. */

#include <malloc.h>

int main(int argc, char *argv[])
{
  int * pi;
  float * px;

  pi = (int *) malloc(sizeof(int));
  px = (float *) malloc(sizeof(float));

  return 0;
}
