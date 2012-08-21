/* Representation of calling context for argv */

#include <stdio.h>

int main(int argc, char * argv[]) 
{
  int i = 0;
  char * s0 = *argv;
  for(i=0;i<argc;i++) {
    printf("%s\n", s0);
    argv++;
    s0 = *argv; // to obtain the result of argv++
  }
  return 0;
}
