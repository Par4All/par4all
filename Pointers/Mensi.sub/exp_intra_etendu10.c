#include<stdlib.h>
#include<stdio.h>

typedef struct LinkedList{
  int *val;
  struct LinkedList *next;
} list;

list* initialize()
{
  int i = 0, *pi = &i;
  list *l = NULL, *nl;
    while(!feof(stdin)){
      nl = malloc(sizeof(list));
      nl->val = pi;
      nl->next = l;
    }
  return l;
}
