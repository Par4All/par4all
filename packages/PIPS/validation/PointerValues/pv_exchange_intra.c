
// stdio.h is not included
#define NULL ((void *) 0)

int main()
{
  int *p, *q, *r;
  int a, b, c;

  p = &a;
  q = &b;

  r = q;
  q = p;
  p = r;
  r = (int *) NULL;

  return 0;
}
