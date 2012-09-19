/* Check pointer arithmetic
 *
 * The correct answer before return is p->a[1] and not p->a[1][0]
 *
 */

int main()
{
  int n = 4, m = 3;
  int a[n][m];
  int (*p)[m] = a;
  p += 1; 
 
  /* The correct answer before return is p->a[1] and not p->a[1][0] */
  return 0;
}
