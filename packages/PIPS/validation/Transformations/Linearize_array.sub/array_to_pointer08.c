
#define NCELL 1024

typedef double fftw_complex[2];
typedef struct { float a; float b; } mystruct;

int main ( int argc, char **argv ) {
  fftw_complex cdens[NCELL];
  mystruct m[NCELL];
  int i;
  for(i=0; i<NCELL; i++) {
    cdens[i][0]=1;
    cdens[i][1]=0;

    m[i].a = 1;
    m[i].b = 0;
  }
  return 0;
}



