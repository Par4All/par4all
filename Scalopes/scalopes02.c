#define SIZE 100

int main (int* argc, char**  argv) {
  int i, j;
  int a[SIZE];

  for (i = 0; i < SIZE; i++)
    for (j = 0; j < SIZE; j++)
      a[j] += 1;

  return 0;
}
