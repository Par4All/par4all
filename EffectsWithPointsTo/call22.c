/* To debug issues linked to arrays passed as parameters... because
 * they cannot be handled like other kinds of parameters.
 *
 * Excerpt from call02.c
 */


void call22(int * q[10])
{
  *(q[4])=3;
}

int main()
{
  int a[10];
  int * ap[10];
  int i;

  for(i=0;i<10;i++)
    ap[i] = &a[i];

  call22(ap);
  return 0;
}
