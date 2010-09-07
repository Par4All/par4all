int main () {
  int result = 0;
  int a[100];
  int i = 0;

  for (i=0; i<100; i++) {
    a[i] = i;
    result += i;
  }

  return result;
}
