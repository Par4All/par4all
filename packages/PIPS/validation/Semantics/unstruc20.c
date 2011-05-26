
#include <stdlib.h>
#include <stdio.h>

int alea(void) {
	return rand() % 2;
}

void unstruc20(int n)
{
  // Like unstruc01, but does not show bug on non-exiting unstructured
  // by Nga Nguyen because the unreachable code, the second printf,
  // seems to be eliminated by the controlizer.

  int i = 0;

 l100:
  if(alea())
    i++;
  if(i>=n) exit(0);
  printf("%d\n", i);
  goto l100;

  printf("%d\n", i);
}
