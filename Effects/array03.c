#include <stdlib.h>
#include <stdio.h>

#define N 5
#define M 3

float d[N][M];

int foo(float (*b)[M])
{
  float c;
  (*b)[3] = 2.0;
  c = (*b)[3];
  b[1][3] = 2.0;
  c = b[1][3];

  ((*b)[3])++;
  (*b)[3] += 5.0;
  (b[1][3])++;
  b[1][3] += 5.0;

  return (1);
}

int foo2(float b[N][M])
{
  float c;
  (*b)[3] = 2.0;
  c = (*b)[3];
  b[1][3] = 2.0;
  c = b[1][3];

  ((*b)[3])++;
  (*b)[3] += 5.0;
  (b[1][3])++;
  b[1][3] += 5.0;

  return 1;
}

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
  float a[N][M], ret;

  ret = foo(a);
  ret = foo2(a);
  ret = foo3();

  return 1;
}
