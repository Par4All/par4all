void tridag(a,b,c,r,u,n)
float a[],b[],c[],r[],u[];
int n;
{
	int j;
	float bet,*gam,*vector();
	void nrerror(),free_vector();

	gam=vector(1,n);
	if (b[1] == 0.0) nrerror("Error 1 in TRIDAG");
	u[1]=r[1]/(bet=b[1]);
	for (j=2;j<=n;j++) {
		gam[j]=c[j-1]/bet;
		bet=b[j]-a[j]*gam[j];
		if (bet == 0.0)	nrerror("Error 2 in TRIDAG");
		u[j]=(r[j]-a[j]*u[j-1])/bet;
	}
	for (j=(n-1);j>=1;j--)
		u[j] -= gam[j+1]*u[j+1];
	free_vector(gam,1,n);
}
