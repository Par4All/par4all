// want to be sure dead code elimination doesn't delete declaration
// initialization in declaration and proper effect compute

int use_def_elim16c_graph()
{
  int r=0;

  r = 1;
  r = 0;

  return r;
}
