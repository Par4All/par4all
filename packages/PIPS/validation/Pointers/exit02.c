/* Check the impact of exit() */
#include<stdlib.h>
void exit02()
{
  int i;
  int * j;
  i = 1;
  if(i){
    j = &i;
    exit(0);
  }
  else {
    static int *p;
    p = malloc(i*sizeof(int));
  }

  i = 2;
}
