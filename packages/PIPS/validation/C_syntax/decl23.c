// Reference to a function

void decl23a()
{
}

void decl23()
{
  struct {
    void (*foo)();
  } s;
  //s.foo = decl23a;
  s.foo = decl23; //decl23 is not known yet by the parser
}
