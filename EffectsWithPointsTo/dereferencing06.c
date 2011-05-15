void dereferencing06()
{
  double x = 1.;
  double * p = &x;

  /* This read has no effect unless p == 0 and the compiler does not optimize? */
  *p++;
  *p++=1;
}

main()
{
  dereferencing06();
}
