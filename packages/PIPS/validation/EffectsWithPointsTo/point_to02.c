/* p was not initialized...
 *
 * Instead of keeping yet another buggy piece of code, I use c to
 * initialize p.
 */

void point_to02()
{
  struct s {
    int a;
    int b[10];
  } c, *p;

  p = &c;

  p->a = 1;
  p->b[2] = 3;
}
