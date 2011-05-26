// conditional operator
int main()
{
  int *p, *q, *r;
  int a = 0, b = 1, c = 2, d = 4;

  p = &a;
  q = (a == b)? &c: &d;
  r = (a<b) ? &c: q;
  return(0);
}
