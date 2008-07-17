void splie2(x1a,x2a,ya,m,n,y2a)
float x1a[],x2a[],**ya,**y2a;
int m,n;
{
	int j;
	void spline();

	for (j=1;j<=m;j++)
		spline(x2a,ya[j],n,1.0e30,1.0e30,y2a[j]);
}
