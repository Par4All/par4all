int convold1(int X[13][13],int coeff[6],int i,int j)
{
  int m, res;
  
  /* for  (m = 0; m<= 5; m++) */

  res = 0;
  for  (m = 0; m<= 5; m++)
    res = res + coeff[m]* X[i+m][j];

  return res;
}
