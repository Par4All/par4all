/* Check points-to analysis on a simple list data structure
 *
 * Discussed 11 August 2011
 */

typedef struct list { int * content; struct list * next;} * list;

void recurrence04(list l, list m)
{
  l->next = m;
  return;
}
