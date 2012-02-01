/* Dealing with unsigned as loop increments
 *
 * The sign of ui must be inferred to generate a precondition within
 * the loop.
 *
 * This can be dealt with in two different ways: with a better
 * recognition of the increment sign when analyzing loops or with type
 * information added in statement_to_transformer() and
 * statment_to_postconditon. The first solution is lighter. The secod
 * one has been tried already using the full type information,
 * e.g. unsigned char x; => 0<=x<=255, but the added constraints lead
 * to a complexity explosion. The added information might be
 * restricted to 0<=x.
 */

int main(void)
{
  int i, a[100];
  unsigned int ui;

  for(i=0; i<100; i+=ui)
    a[i]=0;
}
