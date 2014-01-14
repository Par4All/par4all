/* FI: I want all source code in one file for debugging and
 * maintenance. Also, I do not need spurious functions that are of no
 * use for the final results about "main". Hence the inlining.
 *
 * Same as list07.c, but with call to list_free() not commented out.
 */

// #include "list.src/list.c"

#include "list.src/list.h"

#include <stdlib.h>

struct cons_t {
  double value;
  list next;
};

list list_cons(double v, list l)
{
  list n = (list) malloc(sizeof(struct cons_t));
  n->value = v;
  n->next = l;
  return n;
}

void list_free(list l)
{
  list n = l, p = l;
  while (n!=nil)
  {
    n = n->next;
    free(p);
    p = n;
  }
  return;
}

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

int main(void)
{
  list l;
  l = nil;
  l = list_cons(1.0, nil);
  l = list_cons(2.0, l);
  l = list_cons(3.0, l);
  int n = list_len(l);
  list_free(l);
  return n; // to avoid a gcc warning
}
