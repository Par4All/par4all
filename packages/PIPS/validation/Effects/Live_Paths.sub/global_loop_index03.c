// global variable used as loop index
// re-used afterwards in another module not called from the same module
//   -> loop must not be parallelized in foo

#include <stdio.h>
int i;


void foo()
{
  int j,a[10];

  for (i = 0; i <10; i++)
      a[i] = i;

  for (j = 0; j<10; j++)
    printf("a[%d] = %d\n", j, a[j]);

}

void bar()
{
  printf("i=%d\n",i);
}


int main()
{
  foo();
  bar();
  return 0;
}
