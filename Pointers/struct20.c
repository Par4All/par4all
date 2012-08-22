/* Do we exploit references to generate points-to information?
 *
 * See here the call to read_towns() in main(). Same observation for
 * arguments of printf() in other parts of hyantes.
 *
 * It is not clear what should be done with a on-demand approach...
 *
 * Excerpt from hyantes and extension of struct19.c
*/

#include <stdio.h>
#include <stdlib.h>

typedef double town[3];

typedef struct {
    size_t nb;
    town *data;
} towns;

towns read_towns(const char fname[])
{
  FILE * fd = fopen(fname,"r");
  towns the_towns /* = { 1 , malloc(sizeof(town)) }*/ ;
  return the_towns;
}


int main(int argc, char * argv[])
{
  towns t = read_towns(argv[1]);
  return 0;
}
