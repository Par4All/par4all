#include <stdlib.h>

void star() {
}

int main(void)
{
  void (* point)(void);

  /* partial_eval & normalization should cope with function address
     expressions */
  point = star;

  point = &star;

  return point != NULL;
}
