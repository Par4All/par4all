// exemple pour passage de parametres

#include <stdlib.h>
#include <stdio.h>

struct rec_st;
typedef struct rec_st * recursive;
struct rec_st {
  int i;
  recursive n;
};

// construction d'une liste
recursive foo(int i, recursive p)
{
  recursive q;
  q = (recursive) malloc(sizeof(struct rec_st));
  q->i = i;
  q->n = p;
  return q;
}

// copie d'une liste
recursive copy(recursive p)
{
  recursive q, r;
  r = NULL;
  q = p;
  while (q!=NULL) {
    recursive t;
    t = (recursive) malloc(sizeof(struct rec_st));
    t->i = q->i;
    t->n = r;
    r = t;
    q = q->n;
  }
  return r;
}

int main(void)
{
  recursive p, q;
  p = NULL;
  p = foo(1, p);
  p = foo(2, p);
  p = foo(3, p);

  // copie de la liste
  q = copy(p);

  fprintf(stdout, "q->i: %d\n", q->i);
  fprintf(stdout, "q->n->i: %d\n", q->n->i);
  fprintf(stdout, "q->n->n->i: %d\n", q->n->n->i);
  fprintf(stdout, "q->n->n->n: %p\n", q->n->n->n);
  return 0;
}
