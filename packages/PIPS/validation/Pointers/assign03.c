void assign03()
{
  extern char * malloc(int);
  double * p = (double *) malloc(10*sizeof(double));
  int j = 2;

  p[j] = 1.;
}

void foo()
{
  assign03();
}
