void point_to01()
{
  struct s {
    int a;
    int b;
    int c;
    int d;
  } c, *p;

  p = &c;

  if(1) {
    p->a = 1;
    p->b = 2;
    p->c = 3;
    p->d = p->a;
    p->d = (p->a = 4);
  }
}
