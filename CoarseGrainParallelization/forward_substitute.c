
const int size = 1000;


int main () {
  int b[size];
  int i = 0;
  int result = 0;

  for (i=0; i < size; i++) {
    b[i] = i;
  }

  for (i=0; i < 100; i++) {
    int tmp = b[i] * b[i];
    result += tmp;
  }

  for (i=0; i < 100; i++) {
    result += b[i] * b[i];
  }

  return result;
}
