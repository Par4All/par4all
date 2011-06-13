

int main() {
  int A[10][10];
  int i,j;
// This pragma use to lead to an invalid omp code because we don't detect that an omp pragma already exist
#pragma omp parallel for
  for(i=0; i<10; i++) {
    for(j=0; j<10; j++) {
      A[i][j] =0;
    }
  }

}
