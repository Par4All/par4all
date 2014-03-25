/* Test if the region of a 1-D tile is precise
 *
 * This tile02, 03, 04 and 05 serie was started to see if tiled code
 * could be analyzed precisely
 *
 * Issues here:
 *
 * 1. The external loop is not recognized as a DO loop because ts
 *  could be modified in the loop body
 *
 * 2. The inner loop is not recognized as a DO loop because the
 * condition does not look like an upper bound. Further pattern
 * matching capability and a min intrisics are necessary.
 *
 * 3. The MUST/EXACT loop analysis is not able to cope with a MIN
 * function as upper bound. Nor with a MAX.
 *
 * 4. WHILE loops are not analyzed as well as they could to find exact
 * regions. It would be very close to pattern matching DO loops, but
 * in convex array region analysis.
 *
 */

void tile02(int n, int ts, int a[n])
{
  int ti, i;
  for(ti=0;ti<n; ti += ts)
    for(i = ti; i < ti+ts && i < n; i++)
      a[i]=0;
}


