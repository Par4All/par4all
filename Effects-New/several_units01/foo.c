#include "types.h"

int foo(my_type t)
{
  t.a = 0;
  return t.a;
}
