#include <stdio.h>

// Check that scalarization won't bypass the guard, 
// generating an out of bound access
int guard(int i, int n, int a[n], int b[n]) {
  if(i<n) {
    b[i] = a[i]/2 + a[i]/3;
  }
}

int main(int argc, char **argv) {
  int n=100;
  int a[n],b[n];
  
  // Ensure region IN for guard
  for(int i=0; i<n; i++) {
    a[i]=i; 
  }  

  // i goes over n ! (avoid precondition optimization)
  for(int i=0; i<n+10; i++) {
    guard(i,n,a,b);
  }
  
  
  // Ensure region OUT for guard
  for(int i=0; i<n; i++) {
    printf("%d\n",b[i]);
  }
  
}
