/* Make sure that y is seen as private variable and that then for
   loop is declared parallel. */

int main () {
  float a[10];
  // comments 1
  int i, j;
  float x = 2.12;
  // float y = 2.0;
  {
    float z = 3.0;
    x += z;
  }
  for (i = 0; i < 10; i++) {
    float y = 2.0;
    a[i] = x*y;
  }
}
