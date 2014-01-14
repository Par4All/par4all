/* To investigate Example 5.4 on Page 13 of INRIA TR 7560
 *
 * Their four clause invariant is encoded with only three constraints
 * by PIPS, without making a difference between a parameter and a
 * variable.
 *
 * The impact of R+ wrt to R* is shown using z.
 */

int main()
{
  int x, x0, y, y0, n;

  x = x0, y = y0;

  if(n>=2) {
    float z;
    while(z>0.) {
      x++, y+=1-n;
      z-=1.;;
    }
    z = 0;
  }
  return 0;
}
