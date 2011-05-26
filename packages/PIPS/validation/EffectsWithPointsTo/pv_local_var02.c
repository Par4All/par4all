/* to test the elimination of local variables */
int main()
{
  int i;
  int *p;
  int a[10];

  for(i = 0; i<10; i++)
    {
      int *q;
      q = & a[i];
      p = q;
    }

  return(0);
}
