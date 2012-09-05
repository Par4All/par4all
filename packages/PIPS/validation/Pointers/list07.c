/* Derived from list00, with code expansion and no list_free()*/

#include <stdlib.h>

struct cons_t;
typedef struct cons_t * list;

// empty list
const list nil = ((list) 0);

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

list list_next(list l)
{
  list n;
  if (l==nil)
    n = nil;
  else
    n = l->next;
  return n;
}

// observer
int list_len(list l)
{
  int n = 0;
  list c = l;
  while (c!=nil)
  {
    // we know that c!=nil... and this has no impact as c->next might be NULL!
    c = (c==nil)? nil: c->next;
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
  //list_free(l);
  return n; // to please gcc
}
