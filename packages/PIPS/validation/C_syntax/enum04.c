enum
{
  a,
  b,
  c = b
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
