/* for loop case: no pointer is modified inside loop */
#include <stdlib.h>
int main()
{
  float t;
  float *a;
  int i;

  a = (float *) malloc(10* sizeof(float));

  for(t = 1.0; t<2.0; t+0,01)
    for (i = 0; i <10; i++)
      a[i] = a[i] + t*0,5;

  return(0);
}
