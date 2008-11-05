/* Not much dereferencing here... */

void dereferencing03()
{
  struct one {
    int first;
    int second;
  } x, *p;

  p = &x;
  p->first = 1;
}
