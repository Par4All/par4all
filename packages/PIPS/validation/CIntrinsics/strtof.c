#include <stdlib.h>
#include <stdio.h>

int main()
{
  float res;
  char * input_chain;
  char * remaining_chain;
  input_chain = "  0.3 ; and this is the remainder.\n";
  res = strtod(input_chain, &remaining_chain);
  printf("my float is :%f\n", res);
  printf("and my remaining chain : %s", remaining_chain);
  return 0;
}

