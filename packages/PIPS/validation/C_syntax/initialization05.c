/* Check what happens with a stupid initialization, perfectly legal
   for gcc 4.3.3, and easy to implement (!), as can happen
   with some program transformations.

   No precondition is computed with "int a = a;".
 */

int initialization05()
{
  int a = a;
  int i;

  i = a+1;
  a = 2;
  return i;
}
