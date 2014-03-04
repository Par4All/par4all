// use-def elimination should have no effects
// same than use_def_elim12b but with cumulated effects

int use_def_elim12c()
{
  int r=0;

  if (1)
    r = 1;
  else
    r = 0;

  return r;
}
