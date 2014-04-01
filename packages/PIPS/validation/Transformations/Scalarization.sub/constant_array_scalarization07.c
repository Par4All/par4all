// Check that type mix are OK

void constant_array_scalarization07()
{
  int * pa[3], a[3];
  double * pb[3], b[3];

  pa[0] = &a[0];
  pa[1] = &a[1];
  pa[2] = &a[2];
}
