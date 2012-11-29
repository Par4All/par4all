/*
 *
 * Note: the second loop is not recognized as a do loop because the
 * address of the index "i" is taken. Heuristics in
 * guess_write_effect_on_entity_walker().
 */

#include <stdlib.h>

void malloc02()
{
  int *p[10];
  int i;

  for(i=0; i<10; i++) {
    p[i] = malloc(sizeof(int));
    *p[i] = i;
  }

  for(i=0; i<10; i++) {
    p[i] = malloc(sizeof(int));
    p[i] = &i;
  }
}
