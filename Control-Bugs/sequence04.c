// Debug the destructuration of a C sequence by the new controlizer

// Same as sequence03, but with a declaration in the sub-block

#include <stdio.h>

void sequence04()
{
  int i = 4;

  if((i%2)==0) goto l100;
  i = i + 10;
  {
    int i = 0;
  l100:

    i = i + 20;
  }
  printf("i=%d\n", i);
  return;
}

main()
{
  sequence04();
  return 0;
}
