
#include<stdlib.h>

void pointer08()
{
  struct tree{
    struct tree *tree_list[10];
  };
  int i;
  struct tree *t = (struct tree*) malloc(sizeof(struct tree));
  for( i = 0; i<10; i++ ){
    t->tree_list[i] = (struct tree*) malloc(sizeof(struct tree));
  }
  // FI: without this statement, we do not get points-to information
  // for "return"....
  i++;
  return;
}
