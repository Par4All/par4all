int size = 1000;

int main () {
  int b[size];
  int i = 0;
  int result = 0;

  for (i=0; i < size; i++) {
    b[i] = i;
  }

  for (i=0; i < size; i++) {
    result += b[i] * b[i];
  }

  return result;
}
