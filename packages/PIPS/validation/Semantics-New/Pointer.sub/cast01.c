
int main()
{
  float *p, i=0;
  double *q, j=1;
  
  p=&i;
  q=&j;
  q = (float *) p;
  
  return 0;
}
