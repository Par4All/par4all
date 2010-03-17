/* Make sure that redeclaration of a user function does not break the
   previous declaration*/

void decl47a()
{
  int foo(void);
  int i;

  i = foo();
}

void decl47b()
{
  int foo(void);
  int i;

  i = foo();
}
