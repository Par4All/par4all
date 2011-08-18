#include <stdio.h>
void c_clean_declarations03(int x)
{
  // Hide the formal parameter
  // this is illegal C!
  int x;
  int y;

  if(x) {
    int z;
    float r =1.2;
    x+=(int)r;
  }
  fprintf(stderr, "x=%dn", x);
} 
