



int main() {
  int n = rand();
  int i,j;
  int a[n][n];

  // The outer loop is parallel, 
  // It shoud be the only one concerned by gpu_loop_nest_annotate
  for(i = 0; i < n; i++) { 
    int j;
    // This is a sequential loop
    for(j = 1; j< n; j++) {
      a[i][j]=a[i][j-1];
    }
    // This is a parallel loop
    for(j = 0; j< n; j++) {
      a[i][j]=j;
    }
  }
}
