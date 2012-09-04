/* Declaration and initialization of struct
 *
 * Plus use of intrinsics in pointer initializations, here, fopen().
 *
 * Excerpt from hyantes
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
  towns the_towns = { 1 , malloc(sizeof(town)) };
  return the_towns;
}
