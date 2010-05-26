/* test of macros from library Alternative spellings <iso646.h> */

#include <stdio.h>
#include <iso646.h>

int main()
{ int a,b,c;

  a=b=1;
  c=0;
  if (a xor c)
  printf("a=%d  xor  c=%d = True",a,c);
  
  if(not c)
  printf ("\nc=False \n");
}
