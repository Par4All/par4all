/* *x = &a; 

 x-> y->a
     y->b
 */
int main() {
  int a = 2, b = 1, **x, *y = &a;
  x = &y;
  *x = &b;
 
  return 0;
}
