


void fuse_reuse(int n, int a[n], int b[n],int c[n]) {
  int i;
  // I may be interested to fuse because of the reuse on a[i]
  for(i=0; i<n; i++) {
    b[i]=a[i];
  } 
  for(i=0; i<n; i++) {
    c[i]=a[i];
  } 
    
}
