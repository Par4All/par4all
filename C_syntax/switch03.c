/* No default case: jump at the end of the switch by default */

#include <stdio.h>

int main()
{
  int i0, i1, i2, i3, id;

  i0 = switch03(0); // 111
  i1 = switch03(1); // 4
  i2 = switch03(2); // 3
  i3 = switch03(3); // 333
  id = switch03(4); // 222
  fprintf(stderr, "i0=%d (111), i1=%d (4), i2=%d (3), i3=%d (333), id=%d (333)\n",
	  i0,i1,i2,i3,id);
}

int switch03(int predicate)
{
  int x = 0;
  switch (predicate) {
  case 0: return 111;
  case 1: x = x + 1;
  case 2: return (x+3);
  case 3: break;
  }
  return 333;
}

