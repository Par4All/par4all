/* Bug: the assignments to root and root->next are buggy: lists are
 * expected but pointers to lists are returned.
 *
 * Please do not fix this code as the type mismatch should be
 * detected. A correct version is available as pml03.c.
 */

#include<stdlib.h>
typedef struct {
  struct list* next;
}list;
int main()
{
  list *root;
  root = malloc(sizeof(list*));
  root->next = malloc(sizeof(list*));
  free(root);
  return 0;
}
