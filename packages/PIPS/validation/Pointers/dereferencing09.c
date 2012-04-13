// to test dereferencing

#include <stdio.h>

int main(){
  int a, aa, *b, *bb, **c;

  a = 1;
  b = &a;
  c = &b;

  // Here c->b->a

  bb = *c;

  // bb has the value of b, hence bb->a

  aa = **c;

  // aa is a copy of a

  return 0;

}
