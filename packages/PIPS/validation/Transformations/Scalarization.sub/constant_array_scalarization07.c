// Check that scopes are not a problem for scalarization

void constant_array_scalarization07()
{
  int a[3];

  a[0] = 1;
  a[1] = 1;
  a[2] = 1;
  {
    int a[3];

    a[0] = 1;
    a[1] = 1;
    a[2] = 1;
  }
}
