/* Second basic test case: new assignments should be generated... at
   the right place if you want to stay C89 compatible */

int split_initializations08()
{
  int i = 1, l = 4;
  // gcc does not allow variable size array direct initialization
  int a[l] = { 1 , 2, 3, i };
  i++;
}
