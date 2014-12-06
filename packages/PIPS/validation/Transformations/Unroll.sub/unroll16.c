/* Check a partial unrolling of a loop containing a typedef statement
 *
 * The loop unroll transformation does too much work, and does it
 * inefficiently in term of memory allocated and wrongly for typedef
 * statements.
 */

void unroll16()
{
  int i, j[10];
 bar:for(i=0;i<10;i++) {
    typedef int mtype;
    mtype x = i;
    j[0]=x;
  }
}
