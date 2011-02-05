void p4a_kernel_launcher_0(float imageout[128][128]) {
  float f;
  f = imageout[1][2];
}

int main() {
  /* P_0 is a pointer to an array. Should be initialized just after, but
     it is not the interest point here */
  float (*P_0)[128][128];

  p4a_kernel_launcher_0(*P_0);

  return 0;
}
