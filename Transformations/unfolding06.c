/* Attempt at reproducing bugs happening in video_survey_freia() */

int unfolding05(int i)
{
  int ret;

  ret = foo(i);
  ret |= bar(i);
}

int main()
{
  int j;

  j = unfolding05(4);
}
