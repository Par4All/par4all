/* Not much value added wrt to point_to01.c */

void point_to03()
{
  struct one {
    int first;
    int second;
  } x, *p;

  p = &x;
  p->first = 1;
}
