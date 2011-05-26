/* Same as use_def_elim02.c, but show scalar OUT regions
 *
 * The OUT regions cannot be used to eliminate all dead code because
 * IN regions of useless statements (i.e. statement with no OUT
 * region) are not empty.
 *
 * Here the definition of i is dead, but it gets an OUT region because
 * i is used to define k, k being useless.
 */
int use_def_elim04(int i)
{
  int j;
  int k;

  j = i + 1;
  i = 2;
  k = i+1;
  return j;
  //return i;
  //return k;
}

int main()
{
  return use_def_elim04(2);
}
