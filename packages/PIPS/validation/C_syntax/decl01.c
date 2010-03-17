/* You can declare a structure before you define it. This is going to
   be tough for the PIPS prettyprinter... */

void decl01(char *dir)
{
  struct c;
  struct c *y;
  struct c {
    int foo;
    double bar;
  } ;
  struct c x;
  x.foo = 1;
}
