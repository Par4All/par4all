
const int size = 1000;


int main () {
  int b[size];
  int i = 0;
  int j = 0;

  for (i=0; i < size; i++) {
    b[i] = 0;
  }

  for (j=0; j < size; j++) {
    for (i=0; i < 100; i++) {
      b[j] = b[j] + i;
    }
  }

  for (i=0; i < 100; i++) {
    for (j=0; j < size; j++) {
      b[j] = b[j] + i;
    }
  }

  return 0;
}
