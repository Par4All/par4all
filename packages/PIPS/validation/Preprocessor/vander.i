void vander(x,w,q,n)
float x[],w[],q[];
int n;
{
	int i,j,k,k1;
	float b,s,t,xx;
	float *c,*vector();
	void free_vector();

	c=vector(1,n);
	if (n == 1) w[1]=q[1];
	else {
		for (i=1;i<=n;i++) c[i]=0.0;
		c[n] = -x[1];
		for (i=2;i<=n;i++) {
			xx = -x[i];
			for (j=(n+1-i);j<=(n-1);j++) c[j] += xx*c[j+1];
			c[n] += xx;
		}
		for (i=1;i<=n;i++) {
			xx=x[i];
			t=b=1.0;
			s=q[n];
			k=n;
			for (j=2;j<=n;j++) {
				k1=k-1;
				b=c[k]+xx*b;
				s += q[k1]*b;
				t=xx*t+b;
				k=k1;
			}
			w[i]=s/t;
		}
	}
	free_vector(c,1,n);
}
