// a list data structure

#include "list.h"

#include <stdlib.h>

struct cons_t {
  double value;
  list next;
};

// alloc/free
list list_cons(double v, list lc)
{
  list n = (list) malloc(sizeof(struct cons_t));
  n->value = v;
  n->next = lc;
  return n;
}

void list_free(list fl)
{
  list n = fl, p = fl;
  while (n!=nil)
  {
    n = n->next;
    free(p);
    p = n;
  }
  return;
}

void list_clean(list * p)
{
  list_free(*p);
  *p = nil;
  return;
}

// getter
// FI: double return...
list list_next(list nl)
{
  //if (l==nil)
  //  return nil;
  //else
  //  return l->next;
  list n;
  if (nl==nil)
    n = nil;
  else
    n = nl->next;
  return n;
}

// FI: same issue as above, double return, would be could tests for unstructured
double list_value(list lv)
{
  /* if (l==nil) */
  /*   return 0.0; */
  /* else */
  /*   return l->value; */
  double v;
  if (lv==nil)
    v = 0.0;
  else
    v = lv->value;
  return v;
}

// observer
int list_len(list ll)
{
  int n = 0;
  list c = ll;
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
