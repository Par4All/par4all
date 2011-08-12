void kernel(int n, int a[n]) {
  a[0] = 0;
}

int main() {
 int i,j;
 int n = 10;
 int a[n];
 int sum;
 for(i=0; i<n; i++) {
    for(j=0; j<n; j++) {
      sum += a[j]+1;
    }
    kernel(n,a);
 }

 int c = a[0];
}


