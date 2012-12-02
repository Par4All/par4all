/* To debug issues linked to arrays passed as parameters... because
 * they cannot be handled like other kinds of parameters.
 *
 * Excerpt from call02.c, same as call22.c, similar to call26.c but ap
 * is initialized in the callee.
 */


void call27(int * q[10])
{
  int i;
  for(i=0;i<10;i++)
    q[i] = (int *) malloc(sizeof(int));
  return;
}

int main()
{
  int * ap[10];

  call27(ap);
  *(ap[4]) = 3;
  return 0;
}
