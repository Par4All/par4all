// basic test statements with scalars
int main()
{
  int *p, *q, *r;
  int a = 0;
  int b = 1;
  int c = 2;
  q = &a;
  r = &b;
  if (1)
    p = q;
  else
    p = r;

  p = &c;
  return(0);
}
