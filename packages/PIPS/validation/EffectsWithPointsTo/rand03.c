
typedef struct {
  float re;
  float im;
} Cplfloat;

#define Pi 3.141592653589793238



void AddNoise(int ny, int nx, Cplfloat pt_in[ny][nx], float Sigma2, Cplfloat pt_out[ny][nx])
{
  int ix, iy;
  double u,A,v,B; 
  
  // MOTIF 
  u=0;
  for(iy=0;iy< ny;iy++)
    {
      for(ix=0;ix< nx;ix++)
	{
	  
	  while(u == 0 || u > 1)
	    u = rand()/32768.;
	  A= sqrt(-2.*log(u));
	  v = rand()/32768.;
	  B= 2.* Pi * v;
	  A=1.0;
	  v=1.0;
	  B=1.0;
	  
	  pt_out[iy][ix].re = pt_in[iy][ix].re + Sigma2*A*cos(B);
	  pt_out[iy][ix].im = pt_in[iy][ix].im + Sigma2*A*sin(B);
	  u = 0;
	}
    }
}



main ()
{
  int ny=10,nx=10;
  Cplfloat pt_in[ny][nx];
  float Sigma2;
  Cplfloat pt_out[ny][nx];
  
  AddNoise(ny,nx,pt_in, Sigma2,pt_out);
}
