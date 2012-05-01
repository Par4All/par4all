// Bug: detect nil as NULL pointer

#include <stdlib.h>

struct cons_t {
  double value;
  struct cons_t * next;
};

typedef struct cons_t * list;

const list nil = ((list) 0);

void list06(list l)
{
  list n = nil;
  list p;
  p = nil;
  if(n!=nil) {
    l = l->next;
  }
  return;
}
