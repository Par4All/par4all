/* To investigate Example 5.4 on Page 13 of INRIA TR 7560
 *
 * I designed this example to kill the simple computation of di and
 * dj.
 *
 * Ideally, I could add it to Vivien's ALICe database and compare
 * results obtained by the different tools...
 */

int main()
{
  int i, j, k, n;

  if(i+j==5) {
    while(k>0) {
      i+=n, j-=n;
      k--;
    }
  }
  return 0;
}
