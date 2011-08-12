/* Same as bool00.c, but different tpips: use a property rather than
   points-to information to get b==1 at return point. */

#include <stdbool.h>

int main(void)
{
  int bla, * pbla;
  //  _Bool b = 1;

  //pbla = &bla;
  // kill all preconditions
  *pbla = 1;

  return 0;
}
