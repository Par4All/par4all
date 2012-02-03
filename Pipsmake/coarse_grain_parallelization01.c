



int main(int argc, char **argv) {
  int i,j;
  int n = atoi(argv[1]);
  int a[n][n];
  
  for(int i=1; i<n;i++) {
    for(int j=i; j<n;j++) {
      a[i][j] = a[i-1][j-1] + a[i][j-1];
    }
  }
  
}



