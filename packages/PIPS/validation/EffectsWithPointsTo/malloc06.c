
/* Check type sensitivity and flow/context-sensitive points-to. Copy of malloc03.c */

#include <malloc.h>

int main(int argc, char *argv[])
{
  int * pi;
  pi = 1+ (int *) malloc(sizeof(int));

  return 0;
}
