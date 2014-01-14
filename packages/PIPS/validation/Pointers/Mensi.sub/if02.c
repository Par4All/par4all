/* Exemple de la section 1.2.4, impact d'un test, chapitre
   interprocedural */

#include<stdio.h>
#include<stdlib.h>

void init(int* p)
{
  if(p == NULL)
    exit(1);
  else
    *p = 0;
}

int main()
{
  int init_p = 1 ;
  int *q = NULL;
  if(init_p)
    q = (int*) malloc(4*sizeof(int));
  init(q);
  return 0;
}
