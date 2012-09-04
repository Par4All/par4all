/* while loop case: pointers assigned inside loop */
#include<stdlib.h>
int main()
{
  int i=0;
  int *p, *q;

  p = (int *) malloc(10* sizeof(int));
  q = p;

  while(i<5)
    {
      *q = i;
      q++;
      i++;
    }

  free(p);

  return(0);
}
