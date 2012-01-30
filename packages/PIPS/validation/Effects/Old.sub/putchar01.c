#include<stdio.h>

int i = 0;

void putchar01()
{
  (int) putchar((int)'a');
  (int) putchar((int)'b');
  (int) putchar((int)'c');
  //i =1;
  //(void) putchar((int)'a');
  /* (void) putchar((int)'\n'); */
}

main()
{
  putchar01();
}
