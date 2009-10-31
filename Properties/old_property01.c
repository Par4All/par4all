/* Check that old properties are used instead of the new ones: the
   loop can be distributed, but none of the resulting loops must be
   parallelized */

int old_property01(void)
{
  int i;
  double t, s=0., a[100];
  for (i=0; i<50; ++i) {
    a[i+50] = (a[i]+a[i+50])/2.0;
    s = s + 2 * a[i];
  }
  return s;
}
