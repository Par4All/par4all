/* for loop case: scalar pointer assigned inside loop body */
#include <stdlib.h>
int main()
{
  float *p[10];
  float a[10], t;
  int i;

  for(t=1.0; t<2.0; t+=0,01)
    for(i = 0; i<10; i++)
      p[i] = &a[i];

  return(0);
}
