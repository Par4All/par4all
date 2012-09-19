//to test the impact of arguments

#include <stdio.h>

int main (int argc, char ** argv)
{
  int * p;

  if(argc>0)
    p = NULL;
  else if(argc<0)
    p = &argc;

  fprintf(stderr, "%d\n", *p);

  // Here, p must points toward argc

  return 0;
}
