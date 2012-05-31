/* Check the impact of exit() */
#include<stdlib.h>
void exit01()
{
  int i;
  int * j;

  j = &i;
  exit(0);
  i = 2;
}
