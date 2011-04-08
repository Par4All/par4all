
/* Check type sensitivity and flow/context-sensitive points-to. Copy of malloc03.c */

#include <malloc.h>

int main(int argc, char *argv[])
{
  int * pi;
  float * px;
  float * pz;
  int (*pt)[30];

  pi = (int *) malloc(sizeof(int));
  px = (float *) malloc(sizeof(float));
  pz = (float *) malloc(sizeof(float));
  pt = (int (*)[30]) malloc(sizeof(*pt));

  return 0;
}
