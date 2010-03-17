/* default case not placed at the end of the switch */

#include <stdio.h>

int main()
{
  int i0, i1, i2, i3, id;

  i0 = switch02(0); // 111
  i1 = switch02(1); // 4
  i2 = switch02(2); // 3
  i3 = switch02(3); // 333
  id = switch02(4); // 222
  fprintf(stderr, "i0=%d (111), i1=%d (4), i2=%d (3), i3=%d (333), id=%d (222)\n",
	  i0,i1,i2,i3,id);
}

int switch02(int predicate)
{
  int x = 0;
  switch (predicate) {
  case 0: return 111;
  default: return 222;
  case 1: x = x + 1;
  case 2: return (x+3);
  case 3: break;
  }
  return 333;
}

