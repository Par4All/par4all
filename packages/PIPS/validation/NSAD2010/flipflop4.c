//   Check analysis of periodic behaviors
//

double g(double x)
{
  return x;
}

int main()
{
  double x[2][10];
  int old = 0, new = 1, i, t;

  t=0;
  while(t<1000) {
    for(i=0;i<10;i++)
      x[new][i] = g(x[old][i]);
    if (new==1)
      new=0,old=1;
    else
      new=1,old=0;
    t++;
  }
}
