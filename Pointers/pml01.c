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
