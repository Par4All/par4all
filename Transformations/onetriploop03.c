# include <stdio.h>

void foo(char * f, int a)
{
}

void onetriploop03(int n)
{
  int i;
  int m;

  m = n;

  for(i = n; i<= m; i++) {
    foo("i = %d\n", i);
  }
  printf("i = %d\n", i);
}

int main()
{
  onetriploop03(10);
}
