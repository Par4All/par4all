#define SIZE 100

int main (int* argc, char**  argv) {
  int i, j, k ,l;
  int a[SIZE];

  for (i = 0; i < SIZE; i++)
    for (j = 0; j < SIZE; j++)
      for (k = 0; k < SIZE; k++)
	a[j] += 1;

  for (i = 0; i < SIZE; i++)
    for (j = 0; j < SIZE; j++)
      for (k = 0; k < SIZE; k++)
	for (l = 0; l < SIZE; l++)
	  a[j] += 1;

  for (i = 0; i < SIZE; i++)
    for (j = 0; j < SIZE; j++)
      a[j] += 1;

  return 0;
}
