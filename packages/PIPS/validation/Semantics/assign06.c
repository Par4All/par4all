/* complex expressions with side effects
 *
 * Basic case of assignment as operator in a logical expression
 */

#include <stdbool.h>

int assign06()
{
  bool j, k;

  j = k = true;

  return j+k;
}
