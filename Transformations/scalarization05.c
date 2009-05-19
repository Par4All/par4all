/* Scalarize an array.
   Expected results:
   a) t[i] should be scalarized
   b) a declaration should be created for the new scalar(s)
*/

int main(int argc, char **argv)
{
  int i, n=100;
  int x[n], y[n], t[n];

  for (i=0 ; i<n ; i++) {
    t[i] = x[i];
    x[i] = y[i];
    y[i] = t[i];
  }
}
