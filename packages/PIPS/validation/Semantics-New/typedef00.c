/* Bug linked to the typedef "boolean" */

typedef enum { false, true } boolean;

boolean foo(void)
{
  return true;
}
boolean bla(void)
{
  return foo();
}
