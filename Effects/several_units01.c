// this is to test that the interprocedural translation type checking
// works even with functions declared in different files and using
// a type declared in a common header file.

#include "types.h"

int main()
{
  my_type t;
  return foo(t);
}
