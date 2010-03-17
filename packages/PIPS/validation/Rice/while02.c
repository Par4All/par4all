/* Check that parallel loops located inside a while foop are parallelized */


void while02(int n)
{
  int i,j;
  float x[100];
  float y[100];

  while(i<n) {
    for(j=0;j<100;j++) {
      x[j] = 0.;
    }
    for(j=0;j<100;j++) {
      y[j] = 0.;
    }
    i++;
  }
}
