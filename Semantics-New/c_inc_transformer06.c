/* simplified version of c_inc_transformer03.c
 *
 * incrementation hidden in a stupid piece of code 
 */


int main(int j, char **unused) {
  int B[10];

  B[j++];

  return j;
}
