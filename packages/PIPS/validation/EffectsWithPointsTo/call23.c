/* To debug issues linked to arrays passed as parameters... because
 * they cannot be handled like other kinds of parameters.
 */


void call23(int * q)
{
  *q=3;
}

int main()
{
  int i, *ip;

  if(i)
    ip = &i;

  call23(ip);
  return 0;
}
