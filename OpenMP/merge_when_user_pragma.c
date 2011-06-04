

int main() {
  int A[10][10];
  int i,j;
// This pragma use to lead to an abort because OMP_MERGE_PRAGMA tried to merge every pragma without looking if they're openmp ones.
#pragma toto
  for(i=0; i<10; i++) {
    for(j=0; j<10; j++) {
      A[i][j] =0;
    }
  }

}
