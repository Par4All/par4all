void kernel(int n, int a[n]) {
  for(int j=0; j<n; j++) {
    a[j] = 0;
  }
}

int main() {
 int i,j;
 int n = 10;
 int a[n]; // Because of the C99 declaration, we are not precise enough ! (see loop04_static.c for a C89 version)
 int sum;
 for(i=0; i<n; i++) {
    a[0]=a[0]+1;
    kernel(n,a);
 }

 int c = a[0];
}


