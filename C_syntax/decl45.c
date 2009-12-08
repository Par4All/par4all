/* Bug encountered in SPEC2000/ammp */

int main()
{
  int i;
  void foo();
  void (*bar)();
  foo(i);
  (*bar)(i);
}
