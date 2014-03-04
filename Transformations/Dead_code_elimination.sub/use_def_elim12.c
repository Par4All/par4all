// use-def elimination should have no effects
int use_def_elim12()
{
  int r;
  int y;

  if (y<1)
    r = y;
  else
    r = 1;

  return r;
}
