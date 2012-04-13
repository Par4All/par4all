/* for loop and pointer arithmetic case, with both pointer lhs and rhs
   modifying assignment
  (inspired from a test case pointed out by SG on 09/14/09) */
int main()
{
  float f;
  int *p, *q;
  int a[10];
  int b[10];

  p = &a[0];
  q = &b[0];
  for(f = 0.0; f<10.5; f+=1.0)
      *p++ = 2 * *q++;

  return(0);
}
