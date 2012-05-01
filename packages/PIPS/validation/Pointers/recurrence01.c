/* Check points-to analysis on a simple list data structure
 *
 * Discussed 11 August 2011
 */

typedef struct list { int * content; struct list * next;} * list;

void recurrence01(list l)
{
  int i = 0;
  l->content = &i;
  l->next->content = &i;
  l->next->next->content = &i;
  l->next->next->next->content = &i;
  return;
}
