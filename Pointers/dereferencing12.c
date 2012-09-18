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

  *p = 1;

  // Here, p must points toward argc

  return 0;
}
