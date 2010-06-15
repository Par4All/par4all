/* Make sure that dereferencements of NULL pointers are detected. Same
   as null pointer_01, but initializations separated from
   declarations. q*/

int main()
{
  int i;
  int * p;
  int ** q;

  p = 0;
  ** q = 0;

  * p = 1;
  * q = &i;

  return 0;
}
