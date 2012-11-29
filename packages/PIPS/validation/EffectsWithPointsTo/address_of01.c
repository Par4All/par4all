/* Use offset and & operator
 *
 * Bug in effects_with_points_to: wrong indexing for "*p"
*/

int address_of01()
{
  int *p, i[10];
  p = 1+&i[0];
  return *p;
}
