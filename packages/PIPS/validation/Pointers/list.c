// a list data structure

#include "list.h"

#include <stdlib.h>

struct cons_t {
  double value;
  list next;
};

// alloc/free
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
}

void list_clean(list * p)
{
  list_free(*p);
  *p = nil;
}

// getter
list list_next(list l)
{
  if (l==nil)
    return nil;
  else
    return l->next;
}

double list_value(list l)
{
  if (l==nil)
    return 0.0;
  else
    return l->value;
}

// observer
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

// setter
list list_set_next(list l, list n)
{
  if (l!=nil)
    l->next = n; // pointer is written
  else
    l = n;
  return l;
}

// no pointer is written
list list_set_value(list l, double v)
{
  // could fail?!
  l->value = v;
  return l;
}
