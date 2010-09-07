/* memmove example : Move block of memory */
#include <stdio.h>
#include <string.h>

int main ()
{
  char str[] = "can be very useful......";
  char str2[40];
  memmove (str2,str+15,11);
  puts (str2);
  return 0;
}
