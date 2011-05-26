/* loop and pointer arithmetic case, with both pointer lhs and rhs modifying assignment
  (inspired from a test case pointed out by SG on 09/14/09) */
int main()
{
  int i;
  int *p, *q;
  int a[10];
  int b[10];

  p = &a[0];
  q = &b[0];
  for(i = 0; i<10; i++)
    *p++ = 2 * *q++;

  return(0);
}
