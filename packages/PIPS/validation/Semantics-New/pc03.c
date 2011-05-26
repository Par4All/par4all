/* Check that encoded controls can be properly analyzed (PC stands for
 * program counter)
 *
 * Same as pc01, but else if are replaced by indepedent if, which
 * should make it harder to analyze. Or may be simpler, because the
 * periodicity is reduced to 1. At least if s0 is initialized to 1.
 *
 *
 * Still works with s0==0 and s1==1
 */

int main()
{
  int s0 = 0, s1 = 1, s2 = 0;
  //int c0 = 1, c1 = 0, c2 = 0;

  while(1) {
    if(s0==1) {
      s0 = 0;
      s1 = 1;
      //c1++;
    }
    if(s1==1) {
      s1 = 0;
      s2 = 1;
      //c2++;
    }
    if(s2==1) {
      s2 = 0;
      s0 = 1;
      //c0++;
    }
  }
  return 0;
}
