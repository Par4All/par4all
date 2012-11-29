/* for loop case: scalar pointer assigned inside loop increment */
#include <stdlib.h>
int main()
{
  float *a, *p;
  int i;

  a = (float *) malloc(10* sizeof(float));

  for(p = a, i=0; i<4; p++)
    *p = 1.0;

  return(0);
}
