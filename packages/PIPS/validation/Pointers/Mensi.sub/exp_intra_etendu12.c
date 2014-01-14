/* A possible example for the section about loops */

#include<stdlib.h>
#include<stdio.h>
#include<stdbool.h>

typedef struct LinkedList{
  int val;
  struct LinkedList *next;
} list;

list * initialize()
{
  list *first = NULL, *previous = NULL;
  bool break_p = false; // added to avoid an untructured...
  while(!feof(stdin) && !break_p){
    list * nl = (list *) malloc(sizeof(list));
    nl->next = NULL;
    if(scanf("%d",&nl->val)!=1)
      break_p = true;
    if(first == NULL)
      first = nl;
    if(previous != NULL)
      previous->next = nl;
    previous = nl;
  }
  return first;
}

int main()
{
  list *res = initialize();

  while(res!=NULL) {
    printf("%d\n", res->val);
    res = res->next;
  }

  return 0;
}
