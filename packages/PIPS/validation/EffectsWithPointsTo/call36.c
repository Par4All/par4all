/* To obtain two read effects on a and b and a warning abount
 * ineffective update of i in call02
 *
 * Simplified version of call02.c. Bug in effect translation.
 */

void call36(int i, int * q[10])
{
  /* This is going to lead to _q_2[*][0] since any pointer points
     implicitly to an array */
  *q[i]=3;
  return;
}

int main()
{
  int a = 1, i;
  int aa[10];
  int * ap[10];

  for(i=0;i<10;i++)
    ap[i] = &aa[i];

  call36(a, ap);
  return a;
}
