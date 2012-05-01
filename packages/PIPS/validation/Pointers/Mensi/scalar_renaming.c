int scalar_renaming()
{
  int *p, i, T, A[100], B[100], C[100], D[100];
  p = &T;
  for(i = 0; i<100; i++) {
    T = A[i] + B[i]; /* S1 */
    C[i] = T + T; /* S2 */
    T = D[i] - B[i]; /* S3 */
    *p = 1; /* S4 */
    A[i] = T*T; /* S5 */
  }

  return T;
}
