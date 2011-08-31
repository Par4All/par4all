/* Bug: partial substitution (submitted by Mehdi Amini)
 *
 * Note: this code should be optimized by loop peeling, by removing
 * the first and the last iterations, and then by removing the dead code
*/

void conditional04(int cols, int jW[cols], int jE[cols])
{
  for (int j=0; j< cols; j++) {
    jW[j] = j-1;
    jE[j] = j+1;
    if(j==0) {
      jW[0] = 0;
    }
    if(j==cols-1) {
      jE[cols-1] = cols-1;
    }
  }
}

int caller(int cols) {
  int jW[cols],jE[cols];

  conditional04(cols,jW,jE);

  return jW[0]+jE[cols-1];

}
