/* *x = *y; 

 x-> x1->a
 y->y1->b
    x1->b
 */
int main() {
  int a = 2, b = 1, **x, **y, *x1 = &a, *y1=&b;
  x = &x1;
  y = &y1;
  *x = *y;
 
  return 0;
}
