/* Check that parallel loops with a privatizable variable are correctly parallelized */


void for04()
{
  int i,k,l,m=0,n=0;
  float x[100];

  /* k is not initialized before the loop */
  for(i=0;i<100;i++) {
      k = i;
      x[k] = 0.;
  }

  /* same loop but l is initialized before the loop */
  l = 0;
  for(i=0;i<100;i++) {
      l = i;
      x[l] = 0.;
  }
  
  /* same loop but m is initialized before the loop and was intialized at the declaration */
  m = 0;
  for(i=0;i<100;i++) {
      m = i;
      x[m] = 0.;
  }

  /* same loop but n is initialized only at the declaration */
  for(i=0;i<100;i++) {
      n = i;
      x[n] = 0.;
  }
}

int main() {
 for04();
}