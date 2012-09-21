//to test the impact of arguments

// Required for NULL
#include <stdio.h>

int main (int argc, char ** argv)
{
  int * p;

  if(argc>0)
    p = NULL;
  else if(argc<0)
    p = &argc;

  // Here, p must point toward undefined, NULL or argc

  *p = 1;

  // Here, p must point toward argc

  return 0;
}
