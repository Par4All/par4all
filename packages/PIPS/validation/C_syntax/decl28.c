/* From Ronan: to see how far we can go in keeping with input source

   Since there is only one internal entity for decl28 we cannot keep
   track of all dummy parameters. However, there is space to memorize
   c as dummy parameter and d as formal parameter. But it does not
   work.
 */

void decl28(int b);
void decl28(int c);
void decl28(int d)
{
  d = 1;
}
