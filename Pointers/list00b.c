/* 
 * Same as list00a.c, but partially unrolled to check the behavior of
 * finite constraints for the points-to relattion.
 *
 * Useful to debug the interprocedural translation
 */

#include "list.src/list.h"

#include <stdlib.h>

struct cons_t {
  double value;
  list next;
};

list list_next(list l)
{
  list n;
  if (l==nil)
    n = nil;
  else
    n = l->next;
  return n;
}

int list_len(list ll)
{
  int n = 0;
  list c = ll;
  c = list_next(c);
  c = list_next(c);
  c = list_next(c);
  c = list_next(c);
  c = list_next(c);
  c = list_next(c);
  c = list_next(c);
  c = list_next(c);
  c = list_next(c);
  c = list_next(c);
  c = list_next(c);
  c = list_next(c);

  return n;
}
