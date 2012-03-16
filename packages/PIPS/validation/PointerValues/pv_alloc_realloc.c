#include <stdlib.h>

int main()
{
  int n = 3;
  unsigned char** p1;
  unsigned char** p2;

  p1 = (unsigned char**) malloc(sizeof(unsigned char *));
  p2 = (unsigned char**) malloc(sizeof(unsigned char *));

  int inc_size = 0;
  inc_size += 10;
  *p1 = (unsigned char*)malloc(inc_size*sizeof(unsigned char));
  *p2 = (unsigned char*)malloc(inc_size*sizeof(unsigned char));
  inc_size += 10;

  for (int i=1 ; i<n ; i++)
    {
      inc_size += 10;
      *p1 = (unsigned char*)realloc(*p1,inc_size*sizeof(unsigned char));
      *p2 = (unsigned char*)realloc(*p2,inc_size*sizeof(unsigned char)); 
    }
  return 0;
}
