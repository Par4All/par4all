/* Check a partial unrolling of a loop containing a typedef statement
 *
 * The loop unroll transformation does too much work, and does it
 * inefficiently in term of memory allocated and wrongly for typedef
 * statements.
 *
 * Use dependent type to make it worse
 */

void unroll18()
{
  int i, j[10];
 bar:for(i=0;i<10;i++) {
    typedef int mtype[i];
    mtype x;
    x[0] = i;
    j[0]=x[0];
  }
}
