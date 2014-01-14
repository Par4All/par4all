/* Third third of array03.c: call to foo3 */

#include <stdlib.h>
#include <stdio.h>

#define N 5
#define M 3

float d[N][M];

int foo3()
{
  float c;
  (*d)[3] = 2.0;
  c = (*d)[3];
  d[1][3] = 2.0;
  c = d[1][3];
  
  ((*d)[3])++;
  (*d)[3] += 5.0;
  (d[1][3])++;
  d[1][3] += 5.0;

  return 1;
}



int main() 
{
  float ret;
  
  ret = foo3();
  
  return 1;
}
