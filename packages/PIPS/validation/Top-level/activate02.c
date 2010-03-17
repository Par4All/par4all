/* Check the tpips behavior. This source code does not really
   matter */

int activate01(void)
{
  int i;
  double t, s=0., a[100];
  for (i=0; i<50; ++i) {
    a[i+50] = (a[i]+a[i+50])/2.0;
    s = s + 2 * a[i];
  }
  return s;
}
