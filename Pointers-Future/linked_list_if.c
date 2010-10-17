/* we transform the while loop into two if instructions */
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
  
  if(!feof(stdin)){
    scanf("%d",&i);
    pi = malloc(sizeof(int));
    *pi = i;
    nl = malloc(sizeof(list*));
    nl->val = pi;
    nl->next = l;
  }
  if(!feof(stdin)){
    scanf("%d",&i);
    pi = malloc(sizeof(int));
    *pi = i;
    nl = malloc(sizeof(list*));
    nl->val = pi;
    nl->next = l;
  }
  if(!feof(stdin)){
    scanf("%d",&i);
    pi = malloc(sizeof(int));
    *pi = i;
    nl = malloc(sizeof(list*));
    nl->val = pi;
    nl->next = l;
  }
  return l;
}
