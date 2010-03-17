#define SIZE 100

int main (int* argc, char**  argv) {
  int i;
  int a[SIZE];

  for (i = 0; i < SIZE; i++) a[i] *= 2;
  for (i = 0; i < SIZE; i++) a[i] += 1;
  return 0;
}
