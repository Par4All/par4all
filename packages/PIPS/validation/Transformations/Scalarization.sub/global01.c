// Test case for the scalarization of global arrays.
// Here a[i] can't be scalarized in the loop because it is accessed
// through a called function, whereas b[i] can be scalarized because
// it is only directly accessed in the loop body (r21279)

#include <stdio.h>
#define MAX 10

int a[MAX];
int b[MAX];


void foo(int k)
{
  a[k] = k;
}

int main()
{
  int res = 0;
  for(int i = 0; i<MAX; i++)
    {
      foo(i);
      b[i] = i*i;
      res = res + a[i] + b[i];
    }
  printf("res = %d\n", res);
}
