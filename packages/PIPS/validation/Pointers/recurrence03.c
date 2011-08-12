/* Check points-to analysis on a simple list data structure
 *
 * Discussed 11 August 2011
 */

typedef struct list { int * content; struct list * next;} * list;

void recurrence03(list l)
{
  l->next->next->next = l->next;
  return;
}
