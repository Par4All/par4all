// Check issue with control simplification, i.e. this piece of code
// should be moved into Transformations

#include <stdbool.h>

int main(void)
{
  bool stabilize = 1;
  int i = 1;

  if(stabilize==1)
    i = 2;

  if(stabilize)
    i = 3;

  return i;
}
