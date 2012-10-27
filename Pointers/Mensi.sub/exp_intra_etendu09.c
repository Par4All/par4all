#include<stdio.h>
typedef struct LinkedList{
  int *val;
  struct LinkedList *next;
}list;

int  count(list *p)
{
  int i = 0;
  if(p != NULL){
    i++; p = p->next;
    if (p != NULL){
    i++; p = p->next;
    }
  }
  return i;
}
