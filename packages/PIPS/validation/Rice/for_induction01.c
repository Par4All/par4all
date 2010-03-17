/* Check that parallel loops with an induction variable are correctly parallelized */

void for_induction01()
{
  int i,k;
  float x[100];
  k=0;

  for(i=0;i<100;i++) {
      k = k + 2;
      x[k] = 0.;
  }
}

