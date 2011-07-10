#include <stdio.h>


void loop_private() {
  int n = 10; 
  int tsteps = 10;
  for (int t = 0; t < tsteps; t++) {
    // Rice should'n break private loop declarations !
    for (int i1 = 0; i1 < n; i1++)
 	printf("toto");
    for (int i1 = 0; i1 < n; i1++)
 	printf("tata");

  }



}
