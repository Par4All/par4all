/* Check that m = m/2; leads to dm<=-1 when m>= 1 as 2*dm ~ m and hence
   dm <= 1.
 */

void divide01()
{
  int m ;
  while(m>1) {
    m = m/2;
    m = m; // to get postcondition
  }
  while(m<-1) {
    m = m/2;
    m = m; // to get postcondition
  }
  return;
}
