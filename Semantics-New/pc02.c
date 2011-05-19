/* Check that encoded controls can be properly analyzed
 *
 * Same as pc01, but encoded with ++ and -- instead of settings to 0
 * and 1; here you get the invariant s0+s1+s2==1, but you lose the
 * upper bound 1 for s0, s1 and s2.
 *
 * The upper bounds are found after transformer refinement
 */

int main()
{
  int s0 = 1, s1 = 0, s2 = 0;
  //int c0 = 1, c1 = 0, c2 = 0;

  while(1) {
    if(s0==1) {
      s0--;
      s1++;
      //c1++;
    }
    else if(s1==1) {
      s1--;
      s2++;
      //c2++;
    }
    else if(s2==1) {
      s2--;
      s0++;
      //c0++;
    }
  }
  return 0;
}
