/* Make sure that dereferencements of NULL pointers are
   detected. */

int main()
{
  int i;
  int * p = 0;
  int ** q = 0;

  * p = 1;
  * q = &i;

  return 0;
}
