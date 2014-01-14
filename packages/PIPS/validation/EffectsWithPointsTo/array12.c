/* FI looking for recursive calls */
/* AM: missing recursive descent in points_to_init_variable(); used to miss. */

int array12(int *p, int *q) {
  int b[*(q=p)];
  return 0;
}


 

