/* Check the impact of anymodule:anywhere on the semantics analysis
 *
 * Discussion between Beatrice and Fabien: meaning of
 * anymodule:anywhere...
 *
 * Same as anywhere01, but the variables are formal parameters
 */

#include <assert.h>

void anywhere02(int n, int *p)
{
  assert(p!=0);

  n = 17;

  /* The write effect on n is absorbed by the unknown write effect
     due to *p =>anymodule:anywhere must imply a write on n */
  *p = 19, n = 2;

  n = 23;

// assuming this is an anywhere write effect (without points-to
  // info), the information on n should be preserved because n is
  // never referenced (i.e. &n does not appear in source code).
  *p = 31;

  // n == 23

  return;
}
