// want to be sure dead code elimination doesn't delete declaration
// no initialization in declaration

int use_def_elim16a()
{
  int r;

  r = 1;
  r = 0;

  return r;
}
