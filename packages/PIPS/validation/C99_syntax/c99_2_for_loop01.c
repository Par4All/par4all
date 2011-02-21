/* Test subsequent declarations of for loops */
int main()
{
  double a, b;

  b = 1;
  a = b*b;

  /* First loop */
  for(int i = 0; i < 67; i++)
    a += b;

  // Second loop
  for(int j = 32; j > -34; j--)
    b -= a;

  /* after the loop */
  return 0;
}
