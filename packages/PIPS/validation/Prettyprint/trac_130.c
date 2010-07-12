/* Make sure that privatized variables prevent loop distribution. */

int main () {
  float a[10];
  float b[10];
  int i, j;
  float x = 2.12;
  float z = 3.0;
  //float y;

  x += z;

  for (i = 0; i < 10; i++) {
    float y = 2.0;
    a[i] = x*y;
    b[i] = z*y;
  }
}
