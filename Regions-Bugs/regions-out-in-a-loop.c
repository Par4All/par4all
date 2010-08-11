



#define SIZE 100000
#define T_MAX 100
#define DT 0.1

void compute( int a[SIZE], int b[SIZE], int c[SIZE]  ) {
  int i;
  for(i=0; i<SIZE; i++) {
    a[i] += b[i]+c[i];
  } 
}

void compute_clone( int a[SIZE], int b[SIZE], int c[SIZE]  ) {
  int i;
  for(i=0; i<SIZE; i++) {
    a[i] += b[i]+c[i];
  } 
}


int main() {
  int a[SIZE],b[SIZE],c[SIZE];
  int i;

  compute(a,b,c);

  for(i = 0; i< T_MAX; i+= DT ) {
    compute_clone(a,b,c);
  }

}
