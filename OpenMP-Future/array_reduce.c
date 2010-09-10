
const int size = 1000;

int main () {
  int a[size];
  int b[size];
  int i = 0, j = 0;
  int result = 0;

  // init arrays
  for (i=0; i < size; i++) {
    a[i] = 0;
    b[i] = i;
  }

  for (j=0; j < size; j++) {
    // should reduce a[j]
    for (i=0; i < size; i++) {
      a[j] += b[i];
    }
  }

  for (i=0; i < size; i++) {
    result += a[i];
  }

  return result;
}
