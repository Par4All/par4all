// use-def elimination should have no effects
// same than use_def_elim13b but with cumulated effects

int use_def_elim13c_graph()
{
  int r=0;

  if (1)
    r = 1;
  if (0)
    r = 0;
  
  return r;
}
