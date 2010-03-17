/* Simplified version of generate07.c to check that local typedefs
   have the proper scope.

   And as a consequence, you can declare an external function... with
   a local type!
 */

void generate13()
{
  typedef union {
    int either;
  } z_t;
  z_t z;
  extern int func(z_t);

  // use an undeclared function without source code, which returns
  // implictly an inta typedef type
  (void) func(z);
}
