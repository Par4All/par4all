typedef int ARRAY[10];

int main() {
  int n = 10;
  int a[n];
  int b[10];
  ARRAY c;

  for(int i=0; i<n; i++) {
    b[i] = i;
    a[i] = i*b[i];
  } 
}
