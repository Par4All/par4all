#include<stdlib.h>

void dereferencing_nowhere01()
{
  int * p;
  int * q;
  if (!(q = malloc(sizeof(int)))) exit(0);
  else p = q;
}
