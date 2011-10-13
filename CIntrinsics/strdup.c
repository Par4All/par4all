/* strdup example : Copy and allocate string */
#include <stdio.h>
#include <string.h>

int main ()
{
  char* str1 = "Sample string";
  char str2[] = "Sample string";
  char* str3;
  str3 = strdup(str1);
  free(str3);
  str3 = strdup(str2);
  free(str3);
  str3 = strdup("popop");
  free(str3);
  printf ("str1: %s\nstr3: %s\n",str1,str3);
  return 0;
}
