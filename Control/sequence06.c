// Debug the destructuration of a C sequence by the new controlizer

// Same as sequence04, but with a C89 code generation

#include <stdio.h>

void sequence06()
{
  int l = -1;
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
}

main()
{
  sequence06();
  return 0;
}
