/* To be out of Emami's patterns */

void assign04()
{
  int *** p;
  int ** q;
  int * r;
  int i;

  r = &i;
  //q = &r;
  p = &q;
  **p = r;
  ***p = 0;
}

void foo()
{
  assign04();
}
