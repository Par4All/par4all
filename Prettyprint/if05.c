// PIPS prettyprinter: no output expected, but the prettyprinted code
// misses key braces and the external else message is printed out.

#include <stdio.h>

int main()
{
  int i, c= 3;

  if(c>2) {
    if(c>4)
      i =1;
  }
  else
    printf("\nexternal else\n");

  return i;
}
