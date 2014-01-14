#include <stdbool.h>

int main(void)
{
  int bla;
  int * pbla;
  _Bool b = 1;

  pbla = &bla;
  
  *pbla = 1;

  return 0;
}
