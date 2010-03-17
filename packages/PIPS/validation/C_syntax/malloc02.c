/* Bug by Beatrice: malloc() cannot be redeclared without an include
   of malloc.h

   Note that the redeclaration is not exact since malloc's signature
   is now: void *malloc (size_t __size) (Ubuntu 0910)
 */

void malloc02()
{
  extern char * malloc();
}
