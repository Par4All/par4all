/* Check logical operators on enum variables
 *
 * No information available for b7 because transformers are not
 * computed in context
 */

#include <stdio.h>
#include <stdlib.h>

typedef enum { false, true } bool;

bool alea(void)
{
  // rand() >=0
  // 0 <= ((>=0) % 2) <= 1
  return rand()%2;
}

int main(void)
{
  bool b1, b2, b3, b4, b5, b6, b7;
  b1 = true;
  b2 = false;
  // b3 in 0..1 because of type? %2?
  b3 = alea();
  // b4 in 0..1 because logical
  b4 = b1 && b3;
  // b5 in 0..1 because logical
  b5 = b1 || b3;
  // b6 in 0..1 because logical
  b6 = !b3;
  // b7 ???
  b7 = b1 ^ b2;

  fprintf(stdout, "b1=%d b2=%d b3=%d b4=%d b5=%d b6=%d b7=%d\n",
          b1, b2, b3, b4, b5, b6, b7);
}
