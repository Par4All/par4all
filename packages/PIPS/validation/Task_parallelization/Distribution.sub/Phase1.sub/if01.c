#include <stdlib.h>

int main() {
  int i,j;
#pragma distributed on_cluster=0
  {
    i=0;
    j=0;
  }
  
#pragma distributed on_cluster=1
  if (rand()) {
    i++;
  }
  else {
    j++;
  }
  
#pragma distributed on_cluster=2
  if (rand()) {
    i++;
  }
  else {
    j++;
  }
  
  return 0;
}