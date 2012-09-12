/* Correct version of pml01.c */

#include<stdlib.h>

// FI: the parser does not generate the proper internal representation
// according to the points-to analysis. The anonymous struct is
// implicitly called "struct list"

//typedef struct {
//  struct list* next;
// } list;

typedef struct list {
  struct list* next;
} list;

int main()
{
  list *root;
  root = malloc(sizeof(list));
  root->next = malloc(sizeof(list));
  free(root);
  return 0;
}
