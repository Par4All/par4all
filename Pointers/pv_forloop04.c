/* for loop case: scalar pointer assigned inside loop increment */
#include <stdlib.h>
int main()
{
  float *p;
  float a[10];
  int i;

  for(i=0; i<10; p = &a[i]);

  return(0);
}
