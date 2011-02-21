/* Test subsequent declarations of for loops with same index name */
int main()
{
  double a, b;

  b = 1;
  a = b*b;

  /* First loop */
  for(int i = 0; i < 67; i++)
    a += b;

  // Second loop same index
  for(int i = 32; i > -34; i--)
    b -= a;

  /* after the loop */
  return 0;
}
