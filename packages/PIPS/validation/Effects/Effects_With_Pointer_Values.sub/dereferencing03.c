void dereferencing03()
{
  double * p;
  int i;

  *(p+1) = 0.;
  *(p+i) = 0.;
}

int main()
{
  dereferencing03();
  return 1;
}

