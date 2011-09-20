// Simplify control: same as if01, but for simplify_control instead of
// simplify_control_directly

#include <stdio.h>

void if02()
{
  int i = 0;

  if(1>2)
    printf("%d\n", i);

  return;
}
