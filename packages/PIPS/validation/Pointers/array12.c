/* FI looking for recursive calls
 *
 * Plus pointer information hidden in array declaration.
 *
 * Bug: The dereferencing of "p" should remove the arc p->NULL before
 * "return 0;"
 */

/* AM: missing recursive descent in points_to_init_variable()*/

int array12(int *p, int *q) {
  int b[*(q=p)];
  return b[0];
}


 

