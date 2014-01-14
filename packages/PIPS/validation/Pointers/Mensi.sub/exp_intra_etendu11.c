/* A possible example for the section about loops
 *
 * The "break" construct leads to a control flow graph. Its analysis
 * leads to a bug.
 */

#include<stdlib.h>
#include<stdio.h>

typedef struct LinkedList{
  int val;
  struct LinkedList *next;
} list;

list * initialize()
{
  list *first = NULL, *previous = NULL;
  while(!feof(stdin)){
    list * nl = (list *) malloc(sizeof(list));
    nl->next = NULL;
    if(scanf("%d",&nl->val)!=1)
      break;
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
