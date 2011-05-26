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
  list *l=NULL, *nl, al;
  l = &al;
  if(!feof(stdin)){
    scanf("%d",&i);
    pi = malloc(sizeof(int));
    *pi = i;
    nl = malloc(sizeof(list*));
    nl->val = pi;
    nl->next = l;
    l = nl;
    nl = nl->next;
    ;;
   if(!feof(stdin)){
    scanf("%d",&i);
    pi = malloc(sizeof(int));
    *pi = i;
    nl = malloc(sizeof(list*));
    nl->val = pi;
    nl->next = l;
    l = nl;
    nl = nl->next;
    ;;
   }
  }
 
  return l;
}
