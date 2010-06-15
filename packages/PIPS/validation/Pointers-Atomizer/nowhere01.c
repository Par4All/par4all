/* Make sure that dereferencements of uninitialized pointers are
   detected. */

int main()
{
  int i;
  int * p;
  int ** q;

  * p = 1;
  * q = &i;

  return 0;
}
