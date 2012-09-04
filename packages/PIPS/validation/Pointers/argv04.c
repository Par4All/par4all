/* Representation of calling context for argv
 *
 * Unless the calling context is built on demand: since argv is not
 * used in argv01, its useful formal context is empty.
 *
 * Code has now been added in argv01 to obtain argv04.c and to force
 * the generation of the IN context.
 *
 * Unfortunately, the handling of loops is not satisfactory. Impicit
 * arrays are also a problem. And the declaration of variable s0
 * modifies the points-to information within the loop...
 */

#include <stdio.h>

int main(int argc, char ** argv) 
{
  int i = 0;
  char * s0 = *argv;
  for(i=0;i<argc;i++, argv++)
    printf("%s\n", *argv);
  return 0;
}
