/* Check that encoded controls can be properly analyzed (PC stands for
 * program counter)
 *
 * Same as pc03, but with calls to alea(). The invariant s0+s1+s2==1
 * is degraded to 1<=s0+s1+s2, s0+s1+s2<=2 as in pc01.c, when the
 * preconditions are computed the first time.
 *
 * The refinement steps explodes: the size of the transformer system
 * explodes because trivially redundant constraints are not
 * eliminated. For instance:
 *
 * s0 <= 5
 * s0 <= 5
 * s0 <= 4
 * s0 <= 4
 * s0 <= 3
 * s0 <= 2
 * 2s0 <= 5
 * 2s0 <= 4
 * ...
 *
 * In the run, this happened within transformer_combine() called from
 * condition_to_tranformer, but it is a cumulative process which you
 * see building up under gdb. A redundancy elimination step is
 * forgotten somewhere, maybe in condition_to_transformer() when
 * preconditions are really used.
 *
 * BUG: Everything works properly the first time with property
 * SEMANTICS_USE_DERIVATIVE_LIST set to true. The transformer
 * refinement does not seem to be useful with that setting.
 */

#include <stdlib.h>

int alea(void)
{
  return rand()%2;
}

int main()
{
  int s0 = 0, s1 = 1, s2 = 0;
  //int c0 = 1, c1 = 0, c2 = 0;

  while(1) {
    if(s0==1 && alea()) {
      s0 = 0;
      s1 = 1;
      //c1++;
    }
    if(s1== 1 && alea()) {
      s1 = 0;
      s2 = 1;
      //c2++;
    }
    if(s2==1 && alea()) {
      s2 = 0;
      s0 = 1;
      //c0++;
    }
  }
  return 0;
}
