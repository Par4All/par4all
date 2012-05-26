/* do while loop case: no pointer is modified inside loop */
#include<stdlib.h>
int main()
{
  int i=0;
  int *p;

  p = (int *) malloc(10* sizeof(int));

  do
    {
      p[i] = i;
      i++;
    } while(i<10);

  return(0);
}
