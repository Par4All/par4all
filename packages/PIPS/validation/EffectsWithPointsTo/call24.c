/* To debug issues linked to an undefined pointer.
 *
 * This piece of code is bugged because "ip" is not initialized. The
 * bug must be dectected and PIPS should not crash.
 */


void call24(int * q)
{
  *q=3;
}

int main()
{
  int i, *ip;

  call24(ip);
  return 0;
}
