/* Test the C99 pragma syntax */

void pragma03()
{
  int i;

  _Pragma("omp parallel")
  {
    int j;

    j = i;
  }

}
