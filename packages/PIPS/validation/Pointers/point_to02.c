void point_to02()
{
  struct s {
    int a;
    int b[10];
  } c, *p;

  p->a = 1;
  p->b[2] = 3;
  return;
}
