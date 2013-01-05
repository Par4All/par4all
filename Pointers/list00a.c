/* FI: I want all source code in one file for debugging and
 * maintenance. Also, I do not need spurious functions that are of no
 * use for the final results about "main". Hence the inlining.
 *
 * Same as list00.c, but limited to list_len() whose proper effects
 * computation core dumps.
 *
 * Will be useful to debug the intraprocedural analysis: just update
 * the corresponding tpips script.
 */

// #include "list.src/list.c"

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

int list_len(list l)
{
  int n = 0;
  list c = l;
  while (c!=nil)
  {
    // we know that c!=nil...
    c = list_next(c);
    n++;
  }
  return n;
}
