void point_to04()
{
  struct s {
    int a;
    int b;
    int c;
    int d;
  } c, *p;

  p = &c;

  p->d = p->a;
  p->d = (p->a = 4);
}
