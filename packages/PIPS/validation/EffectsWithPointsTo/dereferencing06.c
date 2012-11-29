void dereferencing06()
{
  double x[3] = {1., 2., 3.};
  double * p = &x[1];

  /* This read had no effect unless p == 0 and the compiler does not
     optimize? I change the code because buggycode is not needed here
     for EffectsWithPointsTo. */
  x[0] = *p++;
  *p++=1;
  return;
}

int main()
{
  dereferencing06();
  return 0;
}
