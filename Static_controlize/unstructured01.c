#include <stdio.h>

// Foo pass static_controlize
int foo() {

  goto end;
end:  return 0;
}

// bar used to segfault because of declaration of variable l when
// closing the database
int bar() {
  int l = 0;

  goto end;
end:  return l;
}
