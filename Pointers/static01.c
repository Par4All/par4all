/* Implicit initialization of a static pointer
 */

int main()
{
  static int * p;
  int * q;

  q = p;

  return 0;
}
