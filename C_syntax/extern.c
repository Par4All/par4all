extern void sc_to_matrices();
extern int a[10];
extern struct foo { int mem;} x;
extern struct toto;
extern struct foo y;

struct titi;
extern int (*potentials[10])(),(*forces[10])(),nused;

int foo()
{
  extern int i;
  /* Yes, a function may be declared inside itself... */
  extern foo();
  extern (*p)();
  extern (*f)();
  extern int b[20];
}
