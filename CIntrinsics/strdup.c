/* strdup example : Copy and allocate string */
#include <stdio.h>
#include <string.h>

int main ()
{
  char* str1 = "Sample string";
  char* str2;
  str2 = strdup (str1);
  printf ("str1: %s\nstr2: %s\n",str1,str2);
  return 0;
}
