#include<stdlib.h>

typedef struct ListeChainee{
	int* val;
 	struct ListeChainee * next;
}liste;
int count(liste* p)
{
  int i = 0, j, tab[20];
  for(j=0; j<20; j++){
    tab[j] = j;
  }
  while(p != NULL){
    p->val = &tab[i];
    p = p->next; i++;    
  }
  return i;
}
