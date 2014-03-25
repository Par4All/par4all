// want to be sure dead code elimination doesn't delete declaration
// initialization in declaration

int use_def_elim16b_graph()
{
  int r=0;

  r = 1;
  r = 0;

  return r;
}
