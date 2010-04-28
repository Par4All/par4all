// both statements should have the same effects.
#include <stdlib.h>
#include <stdarg.h>
#include <stdio.h>

int main()
{
  (void) printf("coucou\n");
  (void) fprintf(stdout, "coucou\n");
return 0;

}

