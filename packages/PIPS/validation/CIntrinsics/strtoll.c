#include <stdlib.h>
#include <stdio.h>

int main()
{
  long long int res;
  char * input_chain;
  char * remaining_chain;
  input_chain = "  3 ; and this is the remainder.\n";
  res = strtoll(input_chain, &remaining_chain, 0);
  printf("my integer is :%lld\n", res);
  printf("and my remaining chain : %s", remaining_chain);
  return 0;
}

