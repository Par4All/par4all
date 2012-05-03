/* Impact of a function returning a pointer */

int i, j;

int * call22(int k)
{
  int * p = k? &i : &j;
  return p;
}

main()
{
  int * q;

  q = call22(1);
  return 0;
}
