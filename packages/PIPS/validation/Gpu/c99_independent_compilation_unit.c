#include <stdio.h>
extern int atoi(const char *nptr);


int main(int argc, char **argv) {
 int n = atoi(argv[1]);
 int a[n];
 int b[n];


 for(int i =0; i<n; i++) {
   a[i]=i;
 }

 
}
