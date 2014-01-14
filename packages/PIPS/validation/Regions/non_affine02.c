/* Check analysis of non-affine subscript expressions
 *
 * Here, the transformer for k=n*i+4 is more accurate than the region
 * for B[n*i+4]. See comment below.
 */

#define TAILLE 500

void non_affine02(int A[TAILLE], int B[TAILLE], int C[TAILLE][TAILLE], int i, int k)
{
  int n = 10;

  /* Note that the transformer for k is more accurate than the region
     for B[i*i+4] because the region is not dense and because the
     constraints on i are nevertheless propagated. xs*/

  for (i = 2; i < TAILLE/50; i++) {
    B[n*i+4] = 12345;
    k = n*i+4;
  }
}
