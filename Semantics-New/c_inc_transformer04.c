/* simplified version of c_inc_transformer03.c
 *
 * Bug: double increment of j because B is integer. Does not happen
 * with double arrays.
 */


int main(int j, char **unused) {
  int B[10];

  B[j++] = 0;

  return j;
}
