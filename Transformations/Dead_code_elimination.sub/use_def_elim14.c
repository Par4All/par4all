// use-def elimination should have no effects
// same that use_def_elim12
// but with braces to be sure it change nothing for the analysis

int use_def_elim14()
{
  int r;

  if (1)
  {
    r = 1;
    r = r;
  }
  else
  {
    r = 0;
    r = r;
  }

  return r;
}
