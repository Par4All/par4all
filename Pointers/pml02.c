#include<stdlib.h>
#include<stdio.h>
 struct foo {
  struct foo* next;
  int val;
};
typedef struct foo MyStruct;
int main()
{
  MyStruct *root;
  root = malloc(sizeof(MyStruct*));
  root->next = malloc(sizeof(MyStruct*));
  free(root);
  (root->next)->val = 1;
  printf("val=%d",root->next->val);
  return 0;
}
