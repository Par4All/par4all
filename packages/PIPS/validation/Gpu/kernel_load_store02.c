/* Test kernel_load_store on a scalar modification.
*/

#include <stdlib.h>

enum { N = 100 };

void give() {
  int k;
  double array[N];

  for(k = 0; k < N; k++)
    array[k] = 0;

  exit(k);
}
