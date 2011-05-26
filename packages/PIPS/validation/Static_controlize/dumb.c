#include <stdio.h>

// Foo pass static_controlize
int foo() {

  goto end;
end:  return;


}

// bar used to segfault
int bar() {
  int l;

  goto end;
end:  return;


}
