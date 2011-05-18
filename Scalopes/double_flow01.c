/*
* Test Bufferization within a main loop of two differents flow and test their value on each iteration outside kernels.
*/
#include <stdio.h>
#include <stdbool.h>
enum { N = 100};
const int nb_tour = 10;

void init (int size, int tab[size], int value) {
  int i;
  for(i=0; i<N; i++){
    tab[i] = value;
  }
}

bool check (int size, int tab[size], int value) {
  bool result = true;
  int i;
  for(i=0; i<N; i++){
    result &= (tab[i] == value);
  }
  return result;
}

int main() {
  int i,j;

  int a[N];
  int f[N];

  init (N, a, 0);
  init (N, f, 2);
  if (check (N, a, 0) == false) {
    printf ("check init a failed\n");
    exit (1);
  }
  if (check (N, f, 2) == false) {
    printf ("check init f failed\n");
    exit (1);
  }

  for (i = 0; i < nb_tour; i++) {

    for(j=0; j<N; j++){
      a[j]+=3;
      f[j]+=3;
    }

    for(j=0; j<N; j++){
      a[j]-=1;
      f[j]-=1;
    }

    a[0]=a[0]++;

    for(j=0; j<N; j++){
      a[j]-=1;
      f[j]-=1;
    }

    if (check (N, a, i+1) == false) {
      printf ("loop %d : check failed\n", i);
      exit (1);
    }
    if (check (N, f, i+3) == false) {
      printf ("loop %d : check failed\n", i);
      exit (1);
    }

  }

  if (check (N, a, nb_tour) == false) {
    printf ("final check a failed\n");
    exit (1);
  }
  if (check (N, f, nb_tour+2) == false) {
    printf ("final check f failed\n");
    exit (1);
  }

  printf ("success\n");

  return 0;
}


