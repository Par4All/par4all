
#define NCELL 1024

void atp( float *cdens[2] ) {
  int i;
  for(i=0; i<NCELL*NCELL*NCELL; i++) {
    cdens[i][0]=1;
    cdens[i][1]=0;
  }

}

int main ( int argc, char **argv ) {
  float cdens[NCELL][2];
  atp(cdens);
}



