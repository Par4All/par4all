void parallel03()
{
  double x[10];
  double y[10];
  int i;

  y[0] = 0.;
  for(i=1; i<10; i++) {
    x[i] = (double) i;
    y[i] = y[i-1] + x[i];
  }
}
