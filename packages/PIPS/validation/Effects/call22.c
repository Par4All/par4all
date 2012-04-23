// to test the impact of a cast in an actual parameter expression

#define ABS_int(out, in) out = abs (in);

void foo (int size, int out[size], int in[size]) {
  int j;
  for (j = 0; j < size; j++) {
    ABS_int (out[j], in[j]);
  }
}


int main (int argc, char** argv) {
  int size = 5;
  int in1[size][size][size];
  int out1[size][size][size];
  int in2[size][size][size];
  int out2[size][size][size];
  int i = 0, j = 0, k = 0, l = 0;
  for (i = 0; i < size; i++) {
    for (j = 0; j < size; j++) {
      for (k = 0; k < size; k++) {
        in1 [i][j][k] = i - j - k;
        in2 [i][j][k] = i - j - k;
      }
    }
  }

  for (i = 0; i < size; i++) {
    for (j = 0; j < size; j++) {
      for (k = 0; k < size; k++) {
        foo (1, (int (*))&out1[i][j][k], (int (*))&in1[i][j][k]);
      }
    }
  }

  return 0;
}
