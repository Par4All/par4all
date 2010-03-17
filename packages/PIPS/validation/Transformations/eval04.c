/* Check that loop bounds are properly partially evaluated.
 *
 * This seems not to work in Demo-2009/convol-unroll-C.tpips: multiple bugs!
 */

void eval04()
{
  int i;
  int j;

  i = 3 + 4 + 0;

  for(i=1+2+0; i<511-1+0; i++ + 0)
    j = 2;
}
