/* Second basic test case: new assignments should be generated... at
   the right place if you want to stay C89 compatible */

int split_initializations07()
{
    int a[2][3] = { { 1,2,3},{4,5,6}};
    int i;
    i++;
}
