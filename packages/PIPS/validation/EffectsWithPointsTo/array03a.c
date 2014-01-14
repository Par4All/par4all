/* First third of array03.c: foo */

#define N 5
#define M 3

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

  return 1;
}

int main() 
{
  float a[N][M], ret;
  
  ret = foo(a);
  
  return 1;
}
