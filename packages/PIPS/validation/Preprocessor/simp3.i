void simp3(a,i1,k1,ip,kp)
int i1,k1,ip,kp;
float **a;
{
	int kk,ii;
	float piv;

	piv=1.0/a[ip+1][kp+1];
	for (ii=1;ii<=i1+1;ii++)
		if (ii-1 != ip) {
			a[ii][kp+1] *= piv;
			for (kk=1;kk<=k1+1;kk++)
				if (kk-1 != kp)
					a[ii][kk] -= a[ip+1][kk]*a[ii][kp+1];
		}
	for (kk=1;kk<=k1+1;kk++)
		if (kk-1 != kp) a[ip+1][kk] *= -piv;
	a[ip+1][kp+1]=piv;
}
