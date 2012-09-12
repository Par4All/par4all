/* Bug with anonymous struct? Problem for the points-to analysis.
 *
 * The type for root->next is "struct list *" instead of "list *"
 *
 * This may show on the symbol table for root to? No, it does not
 * because root is OK, and root->next is not shown obviously. PIPS
 * makes a difference between the anonymous struct used to define the
 * type "list" and the struct of name "list", "struct list".
 */

#include<stdlib.h>

// FI: the parser does not generate the proper internal representation
// according to the points-to analysis. The anonymous struct is
// implicitly called "struct list"

typedef struct {
  struct list* next;
} list;

int pml03()
{
  list *root;
  root = malloc(sizeof(list));
  root->next = malloc(sizeof(list));
  free(root);
  return 0;
}
