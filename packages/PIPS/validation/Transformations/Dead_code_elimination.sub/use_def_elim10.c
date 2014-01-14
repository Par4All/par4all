/* Copy of use_def_elim02, to check dead code elimination with OUT regions */

int use_def_elim10(int i)
{
  int j;
  int k;

  j = i + 1;
  i = 2;
  k = 3;
  return j;
  //return i;
  //return k;
} 

int main()
{
  return use_def_elim10(2);
}
