/* j is an array, or an error is detected
 *
 * &j[0] does not generate a read effect on j since its address is constant
 */

int main() {
  int *i,j[10],k;
  k = 0;
  j[0] = k; 
  i = &j[0];
  i++ ;

  return 0;
}
