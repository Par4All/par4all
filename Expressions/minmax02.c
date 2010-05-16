#include <stdio.h>

/* FI->SG: I do not understand why/how MAX should disappear after
   partial evaluation */

void main()
{
  int al,lo;
  for(lo=0;lo<10;lo++)
    al=MAX(al,lo);
  printf("%d\n",al);
}
