// use-def elimination should have no effects

int use_def_elim12()
{
  int r;

  if (1)
    r = 1;
  else
    r = 0;

  return r;
}
