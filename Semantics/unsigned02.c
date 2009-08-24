//#include <stdio.h>

unsigned int foo(int k)
{
  return k;
}

int bar(unsigned int k)
{
  return 2*k;
}

main()
{
  unsigned int i = 2;
  int j = 4;
  int k = 6;
  int l;

  i++;
  i += j;
  i = foo(k);
  k = bar(i);
  l = foo(k)*bar(i);
  return;
}
