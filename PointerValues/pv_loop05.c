/* loop case: no pointer is modified inside loop */
#include <stdlib.h>
int main()
{
  int i;
  int *p, *q;

  p = (int *) malloc(10* sizeof(int));
  q = p;

  for(i = 0; i<5; i++)
    {
      *q = i;
      q++;
    }

  free(p);

  return(0);
}
