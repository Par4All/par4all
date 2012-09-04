/* Check type sensitivity. Correct version of malloc03.c
 *
 */

#include <stdio.h>
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
  pt = (int (*)[30]) malloc(10*sizeof(*pt));

  long long int i = (long long int) pt;
  printf("%p\n", pt);

  pt++;

  long long int j = (long long int) pt;
  printf("%p, j-i=%d\n", pt, (int)(j-i));

  return 0;
}
