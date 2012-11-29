/* Check type sensitivity.
 *
 * Bug in second malloc removed because it is not needed for EffectsWithPointsTo
 */

#include <malloc.h>

int main(int argc, char *argv[])
{
  int * pi;
  float * px;

  pi = (int *) malloc(sizeof(int));
  px = (float *) malloc(sizeof(float));

  return 0;
}
