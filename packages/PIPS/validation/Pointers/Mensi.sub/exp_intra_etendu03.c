#include<stdlib.h>
typedef struct LinkedList{
  int *val;
  struct LinkedList *next;
}list;

int  count(list *p)
{
  int i = 0;
  while( p != NULL){
    i++; p = p->next;
  }
  return i;
}
