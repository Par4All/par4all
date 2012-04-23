// Bug: generate a stub after accessing a data structure

// The key instruction is "n = n->next;"

// l is dangling on return

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
  //while (n!=nil) {
  // FI: to simplify the analysis...
  while (n) {
    n = n->next;
    free(p);
    p = n;
  }
  return;
}
