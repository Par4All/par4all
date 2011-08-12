void kernel(int n, int a[n]) {
  a[0] = 0;
}

int main() {
 int i;
 int n = 10;
 int a[n];

 for(i=0; i<n; i++) {
    a[0] = a[0]+1;
    kernel(n,a);
 }

 int c = a[0];
}
