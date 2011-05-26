// Debug the destructuration of a C repeat loop by the new controlizer

#include <stdio.h>

void repeat01()
{
  int c = 10;
  do {
    c--;
    if(c % 2 == 0) goto end;
  }   while(c>0);
  c++;
 end:
  printf("The output must be 8: %d\n", c);
  return;
}

int main()
{
  repeat01();
  return 0;
}
