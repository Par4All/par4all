/* Check use of condition in points-to analysis
 *
 * Note that simplify_control could use points-to information here to
 * simplify the test. The returned value would then be known.
 */

int test02()
{
  int i, j;
  int *p = &i;
  int *q = &j;

  if(p==q)
    i = 1;
  else
    i = 2;

  return i;
}
