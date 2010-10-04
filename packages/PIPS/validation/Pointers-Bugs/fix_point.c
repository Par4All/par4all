/*This test case illustrate the computing of points-to relations for
  loops by finding a fix point */
#include<stdio.h>
typedef struct ListeChainee{
 struct ListeChainee * next;
}liste;
int count(liste* p)
{
  liste* q = p;
  int i = 0;
  while(p != NULL){
    i++;
    p = p->next;
  }
  return i;

}
