void splin2(x1a,x2a,ya,y2a,m,n,x1,x2,y)
float x1a[],x2a[],**ya,**y2a,x1,x2,*y;
int m,n;
{
	int j;
	float *ytmp,*yytmp,*vector();
	void spline(),splint(),free_vector();

	ytmp=vector(1,n);
	yytmp=vector(1,n);
	for (j=1;j<=m;j++)
		splint(x2a,ya[j],y2a[j],n,x2,&yytmp[j]);
	spline(x1a,yytmp,m,1.0e30,1.0e30,ytmp);
	splint(x1a,yytmp,ytmp,m,x1,y);
	free_vector(yytmp,1,n);
	free_vector(ytmp,1,n);
}
