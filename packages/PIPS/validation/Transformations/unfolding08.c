/* Attempt at reproducing bugs happening in video_survey_freia().

   Same as unfolding06, but with a filtering option for functions
   foo() and bar() in unfolding08.tpips.
 */

int foo(int i)
{
  return i;
}

int bar(int i)
{
  return i;
}

int unfolding08(int i)
{
  int ret;

  ret = foo(i);
  ret |= bar(i);
}

int main()
{
  int j;

  j = unfolding08(4);
}
