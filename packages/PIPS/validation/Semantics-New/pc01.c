/* Check that encoded controls can be properly analyzed (PC stands for
 * program counter). To avoid convexity issue, one variable is added
 * for each control location instead of a unique program counter.
 *
 * Does not get the invariant s0+s1+s2==1, but 1<=s0+s1+s2<=2
 * The invariant is found with property SEMANTICS_USE_DERIVATIVE_LIST
 * set to true.
 *
 * Note that the transfer function has period 3, where as the maximum
 * encoded periodicity is 2.
 */

int main()
{
  int s0 = 1, s1 = 0, s2 = 0;
  //int c0 = 1, c1 = 0, c2 = 0;

  while(1) {
    if(s0==1) {
      s0 = 0;
      s1 = 1;
      //c1++;
    }
    else if(s1==1) {
      s1 = 0;
      s2 = 1;
      //c2++;
    }
    else if(s2==1) {
      s2 = 0;
      s0 = 1;
      //c0++;
    }
  }
  return 0;
}
