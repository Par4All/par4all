/*This test case illustrate the computing of points-to relations for
  loops by finding a fix point */
#include<stdio.h>
typedef struct ListeChainee{
  int * val; 
  struct ListeChainee * next;
}liste;
int count()
{
  int i = 0, j;
  int tab[10];
  liste* q, *p, p_formal;
  p = &p_formal;
  q = p;
 

  for(j=0;j<10;j++){
    tab[j] = j;
  }
  while(p != NULL){
    p->val = &tab[i];
    p = p->next;
    i++;
  }
  return i;

}
