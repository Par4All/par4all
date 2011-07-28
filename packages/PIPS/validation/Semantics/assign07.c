/* complex expressions with side effects
 *
 * Basic case of assignment as operator in a float expression
 */

int assign07()
{
  float j = 0., k = 0., l;

  j = k = 1.0;

  l = j + k;

  return (int) l;
}
