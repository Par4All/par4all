
int main () {
  int a[5], b[5];
  int i;

  for (i = 0; i < 5; i++) {
    a[i] = b[i] = i;
  }

  return a[b[2]];
}
