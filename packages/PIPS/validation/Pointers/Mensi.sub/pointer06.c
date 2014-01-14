#include<stdlib.h>
void pointer06()
{
  struct tree{
    int val[10];
    struct tree *left;
    struct tree *right;
  };
  int i;
  struct tree *t = (struct tree*)malloc(sizeof(struct tree));
  t->left = (struct tree*)malloc(sizeof(struct tree));
  t->right =  (struct tree*)malloc(sizeof(struct tree));
  struct tree *tl = t->left;
  struct tree *tr = t->right;
  for( i = 0; i<10; i++ ){
    t->val[i] = tl->val[i] + tr->val[i];
  }
  return;
}
