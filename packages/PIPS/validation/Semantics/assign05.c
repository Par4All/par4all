/* complex expressions with side effects
 *
 * Basic case of assignment as operator in a logical expression
 */

#include <stdbool.h>

int assign05()
{
  int j, k;

  j = k = 2;

  return j+k;
}
