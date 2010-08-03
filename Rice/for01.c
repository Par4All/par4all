/* Check that parallel loops located inside a non-convertible for foop
   are parallelized */


void for01(int n)
{
  int i,j;
  float x[100];

  for(;i<n;i++) {
    for(j=0;j<100;j++) {
      x[j] = 0.;
    }
  }
}
