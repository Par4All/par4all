/* bug seen in Transformations/eval.c: how do we guess effects on
   Fortran parameters? How dowe hanfle C differently from Fortran? */

int fx(int nr, int nw)
{
  nw = nr+1;
  return nw;
}

int foo(int j, int i)
{
  i = 10;
  j = 0;
  return fx(i,j);
}
