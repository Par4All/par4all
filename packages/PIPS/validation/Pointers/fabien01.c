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
recursive cons(int i, recursive p)
{
  recursive q;
  q = (recursive) malloc(sizeof(struct rec_st));
  q->i = i;
  q->n = p;
  return q;
}

// copie inverse d'une liste
recursive revcopy(recursive p)
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

// inversion d'une liste (?)
recursive reverse(recursive p)
{
  recursive r = p;
  recursive previous = NULL;
  while (r != NULL)
  {
    recursive next = r->n;
    r->n = previous;
    previous = r;
    r = next;
  } 
  r = previous;
  return r;
}

int main(void)
{
  recursive p, q;
  p = NULL;
  p = cons(1, p);
  p = cons(2, p);
  p = cons(3, p);

  fprintf(stdout,
          "# p 3 items\n"
          "p->i: %d\n"
          "p->n->i: %d\n"
          "p->n->n->i: %d\n"
          "p->n->n->n: %p\n",
          p->i, p->n->i, p->n->n->i, p->n->n->n);

  // copie inverse de la liste
  q = revcopy(p);

  fprintf(stdout,
          "# q=revcopy(p)\n"
          "q->i: %d\n"
          "q->n->i: %d\n"
          "q->n->n->i: %d\n"
          "q->n->n->n: %p\n",
          q->i, q->n->i, q->n->n->i, q->n->n->n);

  // inverse de la liste
  q = reverse(q);

  fprintf(stdout, 
          "# q=reverse(q)\n"
          "q->i: %d\n"
          "q->n->i: %d\n"
          "q->n->n->i: %d\n"
          "q->n->n->n: %p\n",
          q->i, q->n->i, q->n->n->i, q->n->n->n);

  return 0;
}
