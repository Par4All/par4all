#define NCELL 128



//========================================================================
void updateBug(float force[NCELL][NCELL][NCELL]) {
  int i, j, k;
  for (i = 0; i < NCELL; i++) {
    for (j = 0; j < NCELL; j++) {
      for (k = 0; k < NCELL; k++) {
	// This scalar cause GPU_IFY to badly outline the kernel
        int x = 0;
        force[i][j][k] = 0;
      }
    }
  }
}



int main() {
  float force[NCELL][NCELL][NCELL]; // Force for each particle
  updateBug(force);
  return 0;
}





