#define SIZE 501
typedef float float_t;
float_t save[SIZE][SIZE];

int main(int argc, char *argv[]) {
  int i;
  /* Initialize the border of the destination image, since it is used but
     never written to: */
  for(i = 0; i < SIZE; i++)
    save[i][0] = save[0][i] = save[i][SIZE - 1] = save[SIZE - 1][i] = 0;
  return 0;
}
