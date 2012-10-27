#include<stdio.h>
typedef struct LinkedList{
  int *val;
  struct LinkedList *next;
}list;

int  count(list *p)
{
  int i = 0;
  if(0){
    i++; p = p->next;
    if (0){
    i++; p = p->next;
    }
  }
  return i;
}
