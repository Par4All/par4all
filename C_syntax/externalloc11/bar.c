//Checks the memory allocation of extern variables in 3 files
//bar.c: 2 external declarations and one local variable
extern int i;
extern int foobar;
void bar()
{
  int bar;
  foobar = 10;
}
