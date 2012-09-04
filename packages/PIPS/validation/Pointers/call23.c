/* Impact of arguments in a call */

int i, j;

void call23(int * p)
{
  ; // do nothing
  return;
}

main()
{
  int * p = 0;
  call23(p);
  return 0;
}
