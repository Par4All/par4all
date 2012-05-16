// FI: I do not understand this test case ...

#include<stdlib.h>

void pointer09()
{
  struct tree{
    struct tree *suivant;
    int x;
    int y;
  };
  struct tree* ptr[10];
  int j;
  struct tree *t = (struct tree*)malloc(sizeof(struct tree));
  t->suivant = (struct tree*)malloc(sizeof(struct tree));
  for( j = 0; j<10  ; j++){
    t->x = t->y * 5;
    t->y = t->x *6;
  }
}
