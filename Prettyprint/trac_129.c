
int main () {
  float a[10];
  // comments 1
  int i, j;
  float x = 2.12;

  {
    float z = 3.0;
    x += z;
  }
  for (i = 0; i < 10; i++) {
    float y = 2.0;
    a[i] = x*y;
  }
}
