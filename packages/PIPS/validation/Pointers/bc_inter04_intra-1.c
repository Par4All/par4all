// case study with formal parameter aliasing
// used to invalidate substitution analysis

#include <stdlib.h>
#include <stdio.h>


int main(){
  int ***a1;
  int ***a2;
  
  int ** b3;
  int *c4;
  int **l;

  a1 = (int ***) malloc(sizeof(int**));
  *a1 = (int **) malloc(sizeof(int*));
  **a1 = (int *) malloc(sizeof(int));
  
  ***a1 = 1;
  
  a2 = (int***) malloc(sizeof(int**));
  *a2 = *a1;

  b3 = (int**) malloc(sizeof(int*));
  *b3 = (int*) malloc(sizeof(int));
  **b3 = 3;

  c4 = (int *) malloc(sizeof(int));
  *c4 = 4;
  
  l = *a2;
  *a1 = b3;
  *l = c4;

  printf("%d\n", ***a1);
  printf("%d\n", ***a2);

  return 0;
}
