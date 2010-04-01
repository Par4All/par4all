typedef struct {
    float re;
    float im;
  } Cplfloat;

void call13(int nrec, int nsa, int ntt, Cplfloat ptrin[nrec][nsa],Cplfloat ptrout[nrec-ntt+1][ntt][nsa][ntt][nsa])
{
  int i,j1,j2,k1,k2; 
  float R, I;
  
  R = 0.0;
  I = 0.0;

  for(i=0;i<nrec-ntt+1;i++)
    {
     //    MOTIF     
     for(j1=0;j1<ntt;j1++)
	{
	  for(j2=0;j2<nsa;j2++)
	    {
	      for(k1=0;k1<ntt;k1++)
		{
		  for(k2=0;k2<nsa;k2++)
		    {
		      R = ptrin[i+j1][j2].re * ptrin[i+k1][k2].re + ptrin[i+j1][j2].im * ptrin[i+k1][k2].im;
		      I = - ptrin[i+j1][j2].re * ptrin[i+k1][k2].im + ptrin[i+j1][j2].im * ptrin[i+k1][k2].re;
		      ptrout[i][j1][j2][k1][k2].re = R;
		      ptrout[i][j1][j2][k1][k2].im = I;
		    }
		}
	    }
	}
    }
}
