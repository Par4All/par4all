/* to test the non-elimination of static variables */
int a = 0;
int b = 1;
int main()
{

  if (a>1)
    {
      static int *p = &a;
      static int *q;
      q = &b;
      a = a -1;
    }
  else
    {
      static int *r = &a;
      static int *s;
      s = &b;
      a = a +2;
    }

  return(0);
}
