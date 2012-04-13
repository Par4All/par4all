/* FI looking for recursive calls */
/* AM: missing recursive descent in points_to_init_variable()*/

int foo()
{
  int *p, i=1;
  p = 1+&i;
  return 0;
}


 

