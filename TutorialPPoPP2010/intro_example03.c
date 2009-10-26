int foo(int n)
{
  int i;
  double t, s=0., a[100];
  for (i=0; i<n; ++i) {
    t = a[i];
    a[i+n] = t + (a[i]+a[i+n])/2.0;
    s = s + 2 * a[i];
  }
  return s;
}
