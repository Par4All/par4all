void generate02()
{
  int i = 0;
  double x = 1.;

  // use an undeclared function without source code, which returns
  // implictly an int
  func(i, &x);
}
