// use-def elimination should have no effects
// same than use_def_elim12
// but explicit separate if else in 2 if

int use_def_elim13a_graph()
{
  int r;

  if (1)
    r = 1;
  if (0)
    r = 0;
  
  return r;
}
