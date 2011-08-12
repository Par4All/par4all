void kernel(int n, int a[n]) {
  for(int j=0; j<n; j++) {
    a[j] = 0;
  }
}

int main() {
 int i,j;
 int n = 10;
 int a[10]; // Because we know it statically, we are more precise then (see loop04.c for C99 version)
 int sum;
 for(i=0; i<n; i++) {
    a[0]=a[0]+1;
    kernel(n,a);
 }

 int c = a[0];
}


