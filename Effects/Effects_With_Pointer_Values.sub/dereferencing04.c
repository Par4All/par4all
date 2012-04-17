void dereferencing04()
{
  double * p;
  double * q;
  double * r;

  *(p+(q-r)) = 0.;
}

int main()
{
  dereferencing04();
  return 1;
}

