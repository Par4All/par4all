#include<stdlib.h>
#include<stdio.h>

typedef struct LinkedList{
  int val;
  struct LinkedList *next;
} list;

list* initialize()
{
  list *first, *previous, *nl;
  while(!feof(stdin)){
    nl = malloc(sizeof(list));
    scanf("%d",&nl->val);
    if(first != NULL)
      first = nl ;
    if(previous != NULL) {
      previous->next = nl;
      previous = nl;
    }
  }
  return first;
}

int main()
{
  list *res = initialize();
   scanf("%d",&res->val);
   return 0;
}
