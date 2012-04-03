/* *x = y; 

 x-> y->a
     y->b
 */
int main() {
  int a = 2, b = 1, **x, *y = &a, *x1 = &b;
  x = &x1;
  *x = y;
 
  return 0;
}
