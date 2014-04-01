/* Vivien suggested a non affine bounded domain...
 *
 * Here is a case with 8 points in 2-D, same as rotation01 but with
 * assignments intead of updates.
 *
 * The case should be trivial for ASPIC because the number of states
 * is bounded and small.
 *
 * The case is much harder for a transformer-based approach, because
 * the transformations are not as easy to combine as the states. So
 * transformer lists should be used and should be small because only 9
 * different paths are possible when transformers are computed in
 * context. Either no assignment is performed, or a sequence of
 * assignments starts at any of the eight steps.
 *
 * The trick here is that body(s0)==s0 and hence s0 is the loop
 * invariant and then all preconditions are known.
 *
 * More generally, if the number of reachable states by the loop body
 * is bounded and small, the loop invariant should be easy to compute.
 */

void rotation02()
{
  int x = 1, y = 0;

  while(1) {
    if(x==1&&y==0)
      x=2;
    if(x==2&&y==0)
      x=3, y=1;
    if(x==3&&y==1)
      y=2;
    if(x==3&&y==2)
      x=2, y=3;
    if(x==2&&y==3)
      x=1;
    if(x==1&&y==3)
      x=0,y=2;
    if(x==0&&y==2)
      y=1;
    if(x==0&&y==1)
      x=1,y=0;
  }
}
