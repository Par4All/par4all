// Bug: generate a stub after accessing a data structure

// The key instruction is "n = n->next;"

// l is dangling on return

// Same as list04.c, but the property in the tpips script is
// different: strictt typing is not required here so n, l and p might
// be assumed to points towards an array, just like "int *p;" implies
// a vector of int.

#include <stdlib.h>

struct cons_t {
  double value;
  struct cons_t * next;
};

typedef struct cons_t * list;

const list nil = ((list) 0);

void list_free(list l)
{
  list n = l, p = l;
  while (n!=nil) {
    n = n->next;
    free(p);
    p = n;
  }
  return;
}
