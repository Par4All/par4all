/* Check that loop bounds are properly partially evaluated.
 *
 * This seems not to work in Demo-2009/convol-unroll-C.tpips: multiple bugs!
 */

void eval03()
{
  int i;
  int j;

  i = 3 + 4;

  for(i=1+2; i<511-1; i++)
    j = 2;
}
