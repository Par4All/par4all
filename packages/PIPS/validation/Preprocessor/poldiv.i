void poldiv(u,n,v,nv,q,r)
float u[],v[],q[],r[];
int n,nv;
{
	int k,j;

	for (j=0;j<=n;j++) {
		r[j]=u[j];
		q[j]=0.0;
	}
	for (k=n-nv;k>=0;k--) {
		q[k]=r[nv+k]/v[nv];
		for (j=nv+k-1;j>=k;j--)
			r[j] -= q[k]*v[j-k];
	}
	r[nv]=0.0;
}
