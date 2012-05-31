/* while loop case: no pointer is modified inside loop */
#include<stdlib.h>
int main()
{
  int i=0;
  int *p;

  p = (int *) malloc(10* sizeof(int));

  while(i<10)
    {
      p[i] = i;
      i++;
    }

  return(0);
}
