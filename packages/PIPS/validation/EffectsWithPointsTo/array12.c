/* FI looking for recursive calls */
/* AM: missing recursive descent in points_to_init_variable()*/

int foo(int *p, int *q) {
  int b[*(q=p)];
  return 0;
}


 

