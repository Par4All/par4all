
const int size = 1000;

int main () {
  int b[size][size];
  int i = 0;
  int j = 0;

  for (j=0; j < size; j++) {
    for (i=0; i < size; i++) {
      b[i][j] = 0;
    }
  }

  for (j=0; j < size; j++) {
    for (i=0; i < 100; i++) {
      b[j][i] = b[j][i] + i;
    }
  }

  return 0;
}
