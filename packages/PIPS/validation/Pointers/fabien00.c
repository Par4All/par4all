// exemple pour point fixe while

#include <stdlib.h>
#include <stdio.h>

struct rec_st;
typedef struct rec_st * recursive;
struct rec_st {
  int i;
  recursive n;
};

int main(void)
{
  // srand(1);
  int data = 0;
  recursive l;
  l = NULL;
  while (rand()%19) {
    recursive q;
    q = (recursive) malloc(sizeof(struct rec_st));
    q->i = data++;
    q->n = l;
    l = q;
  }
  // affiche la longueur de la liste créée
  fprintf(stdout, "data=%d\n", data);
  return 0;
}
