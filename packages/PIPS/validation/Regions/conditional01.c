/* Check analysis of C conditional operator */

#define TAILLE 500

void conditional01(int A[TAILLE], int B[TAILLE], int C[TAILLE][TAILLE], int i, int k) {
  for (i = 30; i < TAILLE; i++) {
    B[(i%2 == 0)?i:i-1] = 12345;
  }
}
