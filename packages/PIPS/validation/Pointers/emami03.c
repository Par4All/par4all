/* y -> y1->a
   x -> a
 */
int main() {
  int a = 2, *y1 = &a, *x, **y = &y1 ;
  x = *y;
 
  return 0;
}
