/* Check that declarations are not distributed */


void for05(int n)
{
  int j;
  float x[100];

  for(j=1;j<100;j++) {
    int i;
    float y = 0;
    x[j] = x[j-1]+y;
  }
}
