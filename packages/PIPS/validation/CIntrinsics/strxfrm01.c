#include <string.h>
#include <stdio.h>
int main()
{
  char s[]="example";
  char d[10];
  strxfrm(d, s, 3);
  return 0;
}
