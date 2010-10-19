/* How we should compute the fix point for a while loop?*/
#include<stdio.h>
#include<stdlib.h>
typedef struct LinkedList{
  int *val;
  struct LinkedList *next;

}list;

list* initialize()
{
  int *pi, i;
  list *l=NULL, *nl;
  
   while(!feof(stdin)){
    scanf("%d",&i);
    pi = malloc(sizeof(int));
    *pi = i;
    nl = malloc(sizeof(list*));
    nl->val = pi;
    nl->next = l;
    l = nl;
    }

  return l;
}
