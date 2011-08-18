typedef int ARRAY[10];

int vla(int size, int d[size][size], int e[3][3]);

int vla(int size, int d[size][size], int e[3][3]) {
  int n = d[1][1];
  int a[n];
  int b[10];
  ARRAY c;

  for(int i=0; i<n; i++) {
    b[i] = i;
    a[i] = i*b[i];
    d[1][1] = i;
  } 
}
