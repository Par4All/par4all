/* How we should compute the fix point for a while loop?*/
#include<stdio.h>
#include<stdlib.h>
typedef struct LinkedList{
  int *val;
  struct LinkedList *next;

}list;

int  count(list *p)
{
  list *q = p;
  int i = 0;
   while( p != NULL){
     i++;
     p = p->next;
   }

  return i;
}
