/* Attempt at reproducing bugs happening in video_survey_freia().

   Similar to FREIA's case. I assumed that foo() and bar() did not
   need to be defined because I requested an unfolding of unfolding06
   in main, but in fact unfolding is not restricted to
   unfolding06. All call sites to unfolding06 are fully inlined down
   to the smallest function. This make unfolding useless for FREIA
   coarse and middle grain approaches.
 */

int unfolding06(int i)
{
  int ret;

  ret = foo(i);
  ret |= bar(i);
}

int main()
{
  int j;

  j = unfolding06(4);
}
