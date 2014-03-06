// use-def elimination should have no effects
// same than use_def_elim13 but declaration with init

int use_def_elim13b_graph()
{
  int r=0;

  if (1)
    r = 1;
  if (0)
    r = 0;
  
  return r;
}
