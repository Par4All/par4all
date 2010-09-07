



#define SIZE 100000


/* The two following functions are identical, only the call site changes */
void compute_seq( int a[SIZE], int b[SIZE], int c[SIZE]  ) {
  int i;
  for(i=0; i<SIZE; i++) {
    a[i] += b[i]+c[i];
  } 
}

void compute_loop( int a[SIZE], int b[SIZE], int c[SIZE]  ) {
  int i;
  for(i=0; i<SIZE; i++) {
    a[i] += b[i]+c[i];
  } 
}


/* Here we simply call two time the "compute" function sequentially */
void sequential() {
  int a[SIZE],b[SIZE],c[SIZE];
  compute_seq(a,b,c);
  compute_seq(a,b,c);
}


/* Here we call two time the "compute" function, but with a loop */
void loop() {
  float i = 0;
  int a[SIZE],b[SIZE],c[SIZE];
  for(i=0;i<0.2;i+=0.1) {
    compute_loop(a,b,c);
  }
}



int main() {
  loop();
  sequential();
}
