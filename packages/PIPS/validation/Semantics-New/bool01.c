/* Same as bool00.c, but different tpips: use property
   ALIASING_ACROSS_TYPES rather than points-to information to get b==1
   at return point.

   ALIASING_ACROSS_TYPES is not implemented yet because untyped
   anywhere effects are systematically generated.
 */

#include <stdbool.h>

int main(void)
{
  int bla, * pbla;
  _Bool b = 1;

  pbla = &bla;
  // kill all preconditions
  *pbla = 1;

  return 0;
}
