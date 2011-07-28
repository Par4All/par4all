// strtoll with NULL pointer as second argument

#include <stdlib.h>
#include <stdio.h>

int main()
{
  long long int res;
  char * input_chain1, *input_chain2;
  input_chain1 = "  1 ; and this is the remainder.\n";
  res = strtoll(input_chain1, (void *) 0, 0);
  printf("my first integer is :%lld\n", res);
  res = strtoll(input_chain2, NULL, 0);
  input_chain2 = "  2 ; and this is the remainder.\n";
  printf("my second integer is :%lld\n", res);
  return 0;
}

