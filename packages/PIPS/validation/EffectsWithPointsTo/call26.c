/* To debug issues linked to arrays passed as parameters... because
 * they cannot be handled like other kinds of parameters.
 *
 * Excerpt from call02.c, similar to call22.c, but ap is not initialized.
 *
 * So, this piece of code is bugged, but the bug should be detected by
 * PIPS. Currently, it crashes...
 */


void call26(int * q[10])
{
  *(q[4])=3;
}

int main()
{
  int * ap[10];

  call26(ap);
  return 0;
}
