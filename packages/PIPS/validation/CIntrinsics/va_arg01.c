/* va_arg example : This FindMax function takes as its first parameter the amount of additional arguments it is going to get. The first additional argument is retrieved and used as an initial reference for comparison, and then the remaining additional arguments are retrieved in a loop and compared to return the greatest one (which in this case is 892). */
#include <stdio.h>
#include <stdlib.h>

#include <stdarg.h>

int FindMax ( int amount, ...)
{
  int i,val,greater,len;
  va_list vl;
  va_start(vl,amount);
  printf ("\n num=%s \n",vl);
  greater=va_arg(vl,int);
  for (i=1;i<amount;i++)
  {
    val=va_arg(vl,int);
    greater=(greater>val)?greater:val;
  }
  va_end(vl);
  return greater;
}

int main ()
{
  int m;
  m= FindMax (7,702,422,631,834,892,104,772);
  printf ("The greatest one is: %d\n",m);
  return 0;
}
