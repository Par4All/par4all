
/* #include <stdio.h> */


int convold1();
int convold2();
int filtre();
void init_coeff();

int  main()
{
  int X[13][13],Y[8][8],H[8][13],K[8][8];
  int OUT[8][8], COEFF[6], i,j;

  init_coeff(COEFF);

  for (i = 0; i <= 12; i++)
    for (j = 0; j <= 12; j++)
      X[i][j] = (j*(j+i))/2;

  for (i = 0; i <= 7; i++)
    for (j = 0; j <= 12; j++)
      H[i][j] = convold1(X,COEFF,i,j);

  for (i = 0; i <= 7; i++)
    for (j = 0; j <= 7; j++)
      K[i][j] = convold2(H,COEFF,i,j);

  for (i = 0; i <= 7; i++)
    for (j = 0; j <= 7; j++)
      Y[i][j] = filtre(H,K,i,j);

  for (i = 0; i <= 7; i++)
    for (j = 0; j <= 7; j++)
      OUT[i][j] = Y[i][j];

}


void init_coeff(int COEFF[6])
{
  COEFF[0]=1;
  COEFF[1]=-5;
  COEFF[2]=20;
  COEFF[3]=20;
  COEFF[4]=-5;
  COEFF[5]=1;

}

int convold1(int X[13][13],int coeff[6],int i,int j)
{
  int m,res = 0;

  for  (m = 0; m<= 5; m++)
    res = res + coeff[m]* X[i+m][j];

  return res;
}

int convold2(int H[8][13], int coeff[6],int i,int j)
{
  int m, res = 0;

  for  (m = 0; m<= 5; m++)
   res = res + coeff[m]*H[i][j+m];

  return res;
}

int  filtre(int H[8][13],int K[8][8],int i,int j)
{
  int ht,kt,res;

  ht = (H[i][j+2] +16)/32;
  kt = (K[i][j]+512)/1024;
  res = (ht + kt + 1)/2;
  return res;
}
