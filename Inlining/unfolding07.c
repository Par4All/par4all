/* Attempt at reproducing bugs happening in video_survey_freia().

 Same as unfolding06, but with full definition of functions foo() and
 bar().
 */

int foo(int i)
{
  return i;
}

int bar(int i)
{
  return i;
}

int unfolding07(int i)
{
  int ret;

  ret = foo(i);
  ret |= bar(i);
}

int main()
{
  int j;

  j = unfolding07(4);
}
