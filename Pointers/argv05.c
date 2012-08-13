/* Representation of calling context for argv */

#include <stdio.h>

int main(int argc, char * argv[]) 
{
  int i = 0;
  char * s0 = *argv;
  for(i=0;i<argc;i++, argv++)
    printf("%s\n", *argv);
  return 0;
}
