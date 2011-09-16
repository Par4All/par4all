
void cmat_vec_dble(int n, double _Complex (*a)[n][n], double _Complex *v[n], double _Complex (*w)[n])
{

 int i, j;



 for(i = 0; i <= n-1; i += 1) {


    (*w)[i] = 0.0;

    for(j = 0; j <= n-1; j += 1)

      (*w)[i] += ((*a)[i])[j]*(*v)[j];
 }
}
