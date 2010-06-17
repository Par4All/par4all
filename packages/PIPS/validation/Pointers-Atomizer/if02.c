// test case for eval_cell_with_points_to with MAY points-to
#include <stdio.h>

int main() {
  int ***p;
  int a, b, *aa, *bb, **aaa, **bbb;
  
  a = 1;
  b = 2;
  
  aa = &a;
  bb = &b;

  aaa = &aa;
  bbb = &bb;

  p = &bbb;

  if (0==0)
    p = &aaa;
  else
    **p = bb;

  printf("%d\n", ***p);
  
  return 0;
}
