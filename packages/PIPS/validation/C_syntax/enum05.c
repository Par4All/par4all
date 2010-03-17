enum
{
  a,
  b,
  c = b+3
};

main()
{
  int ia;
  int ib;
  int ic;
  int t;

  ia = a;
  ib = b;
  ic = c;

  t = ia + ib + ic;
}
