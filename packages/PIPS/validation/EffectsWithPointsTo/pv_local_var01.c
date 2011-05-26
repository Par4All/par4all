/* to test the elimination of local variables */
int main()
{
  int i = 3;
  int *p;
  int a[10];

  if (i>1)
    {
      int *q;
      q = & a[i];
      p = q;
      i = 0;
    }

  return(0);
}
