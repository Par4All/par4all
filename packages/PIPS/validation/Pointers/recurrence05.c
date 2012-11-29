/* Check points-to analysis on a simple list data structure
 *
 * Discussed 11 August 2011
 */

#include "malloc.h"

typedef struct list { int * content; struct list * next;} * list;

list recurrence05(void)
{
  int i = 0;
  list l = (list) malloc(sizeof(list *));
  l->content = &i;
  l->next = NULL;
  return l;
}
