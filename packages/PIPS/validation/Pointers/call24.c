/* Impact of arguments in a call within a loop */

int i, j;

void call24(int * p)
{
  ; // do nothing
  return;
}

main()
{
  int * p = 0, n;
  while(n) {
    call24(p);
    n--;
  }
  return 0;
}
