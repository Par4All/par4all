#include <stdlib.h>
#include <stdio.h>
#include <string.h>


int main(int argc, char **argv) {
  int n = 1000;
  if(argc>=2) {
    n = atoi(argv[1]); // FIXME error check, use strtol instead
  }

  n=4*n; // To be compatible with int4

  float a[n],b[n],c=32;


  for(int t = 0 ; t<10; t++) {
    for(int i=0; i<n; i++) {
      a[i] = b[i] * c;
    }
    printf("%f\n",a[0]);

    for(int i=0; i<n; i+=4) {
      a[i] = b[i] * c;
      a[i+1] = b[i+1] * c;
      a[i+2] = b[i+2] * c;
      a[i+3] = b[i+3] * c;
    }
  }

  printf("%f\n",a[0]);

  return 0;

}
