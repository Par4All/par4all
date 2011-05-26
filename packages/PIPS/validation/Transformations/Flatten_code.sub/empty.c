#include <stdio.h>

// This simple test used to make flatten code failed (return false, and thus aborting)
void empty(char *name, double a) {
  printf("%s %g\n", name, a);
}

