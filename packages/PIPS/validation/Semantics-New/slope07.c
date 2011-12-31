// Goal: check constraint simplification by function
// small_positive_slope_reduce_coefficients_with_bounding_box()

// Potential bug found in Maisonneuve.sub/whileif_vs_whilewhile

#include <stdio.h>
#include <assert.h>

void slope07()
{
  int x, y, k;

  /* create a bounding box */
  // assert(0<=x && x<=10 && 0<=y && 0<=10);
  assert(0<=x && x<=58);

  if(61*y<=55*x+54) {
    /* x=9, y=9 is reachable */
    /* The new constraints are derived from points (0,0), (9,9),
       (49,45) and (58,53) with slopes 1, 9/10 and 8/9 */
    k = x + y;
  }
  else {
    k = x + y;
  }

  scanf("%d", &x);
  /* create another bounding box */
  assert(0<=y && y<=9);

  if(61*y<=55*x+54) {
    k = x + y;
  }
  else {
    k = x + y;
  }

  return;
}
