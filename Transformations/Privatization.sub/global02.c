// The three loop nests perform the same task; however the first one uses
// function local scalar variable; the second one uses a global variable
// which is passed as a parameter to function foo; the third one uses
// the global variable inside function bar.
// In the second case, the global variable can be privatized by
// privatize_module_even_globals but not by privatize_module.
// In the third case, the global variable should not be privatized as it would require
// cloning foo to isolate the global variable.

int my_global = 10;

void foo (int size, int a[size][size][size][size], int dim0, int dim1, int dim2) {
  int l = 0;
  for (l = 0; l < size; l++) {
    a[dim0][dim1][dim2][l] = 0;
  }

  return;
}

void bar (int a[25][25][25][25], int dim0, int dim1, int dim2) {
  int l = 0;
  for (l = 0; l < my_global; l++) {
    a[dim0][dim1][dim2][l] = 0;
  }

  return;
}

int main (int argc, char** argv) {
  int size = 25;
  int my_local = 10;
  int tab [size][size][size][size];
  int i = 0, j = 0, k = 0;

  for (i = 0; i < size; i++) {
    for (j = 0; j < size; j++) {
      for (k = 0; k < size; k++) {
        my_local = 25;
        foo (my_local, tab, i, j, k);
      }
    }
  }

  for (i = 0; i < size; i++) {
    for (j = 0; j < size; j++) {
      for (k = 0; k < size; k++) {
        my_global = 25;
        foo (my_global, tab, i, j, k);
      }
    }
  }

  for (i = 0; i < size; i++) {
    for (j = 0; j < size; j++) {
      for (k = 0; k < size; k++) {
        my_global = 25;
        bar (tab, i, j, k);
      }
    }
  }


  return 0;
}
