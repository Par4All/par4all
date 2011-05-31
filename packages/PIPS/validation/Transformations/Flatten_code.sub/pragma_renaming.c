

int main() {
  int n = 10;
  int a[n][n];
  int j; // This declaration is here to force flatten_code to rename the j loop indice

  { 
    int i,j; // overwrite previous declaration for j and force flatten code to rename

    // The omp generated clause privatize j, but flatten_code used to "forget" to 
    // rename it while renaming the loop indice
    for(i=0; i<n;i++) {
      for(j=0; j<n;j++) {
        a[i][j]=i*j;
      }
    }
  } 

}
