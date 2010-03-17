/* Bug by Beatrice: malloc() cannot be redeclared without an include
   of malloc.h */

void malloc01()
{
  extern void * malloc();
  void * malloc();
  extern void * foo(int);
  extern void * bar(void);
  extern f();
}
