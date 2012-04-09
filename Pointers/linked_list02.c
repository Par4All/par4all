/* How we should compute the fix point for a while loop? */

/* Unlike linked_list01, the formal parameter p is *not* modified
 * within the function count(), which is much better for debugging!
 */

/* Bug: the iterations do not converge and the lattice is not used to
 * reach a fix point.
 */

#include<stdio.h>
#include<stdlib.h>

typedef struct LinkedList{
  int *val;
  struct LinkedList *next;

}list;

int  count(list *p)
{
  list *q = p;
  int i = 0;
   while( q != NULL){
     q = q->next, i++;
   }
  return i;
}
