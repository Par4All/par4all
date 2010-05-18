#include <assert.h>

int main()
{
  int i;

  i>1? (i = 2): (i = 3);

  i = i>1? 2 : 3;

  assert(i>1);

  return i>1;
}
