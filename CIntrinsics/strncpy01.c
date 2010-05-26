/* strncpy example */
#include <stdio.h>
#include <string.h>

int main ()
{
  char str1[]= "To be or not to be";
  char str2[6];
  strncpy (str2,str1,5);
  str2[5]='\0';
  puts (str2);
  return 0;
}