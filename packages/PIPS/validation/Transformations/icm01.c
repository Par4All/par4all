/* Just a very basic test of Invariant Code Motion (ICM)
 */

int icm01()
{
  int i;
  int j = 3;
  int k;
  int a[10];

  for (i=0;i<10;i++) {
    k = j*j + 1;
    a[i]=k;
  }
}
