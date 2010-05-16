/* Check the type detection when an offset is added to a pointer via
   an array construct. This used to complete the profile of a
   partially declared function.

   Note: this code is not clean because the implicit declaration of
   malloc does not fit the built-in malloc...
 */

void decl30(void)
{
 double foo();
 double (*x)[];
 double z;
 extern void * mymalloc(int);

 x = (double (*)[]) mymalloc(10);
 z = foo(&(*x)[0]);
}
