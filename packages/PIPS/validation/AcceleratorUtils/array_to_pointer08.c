
#define NCELL 1024

typedef double fftw_complex[2];

int main ( int argc, char **argv ) {
  fftw_complex cdens[NCELL];
  int i;
  for(i=0; i<NCELL; i++) {
    cdens[i][0]=1;
    cdens[i][1]=0;
  }
  return 0;
}



