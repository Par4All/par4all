#include <string.h>
#include <stdlib.h>

char* foo()
{
  return (char*)malloc(10);
}

main ()
{
  memcpy(foo(),"toto",4);
}
