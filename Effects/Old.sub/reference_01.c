/* Empty statement with exception: no big deal, gcc seems to
   eliminate them anyway! Even in -O0. */

int reference_01(void)
{
  int i = 0;
  int j = 1;

  // effects on useless expression
  0;
  // effects on empty references...
  j/i;
  (void) j/i;
  return i/j;
}

int main(void)
{
  int k = reference_01();

  return k;
}
