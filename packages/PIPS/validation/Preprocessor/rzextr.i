extern float **d,*x;	/* defined in BSSTEP */

void rzextr(iest,xest,yest,yz,dy,nv,nuse)
int iest,nv,nuse;
float xest,yest[],yz[],dy[];
{
	int m1,k,j;
	float yy,v,ddy,c,b1,b,*fx,*vector();
	void free_vector();

	fx=vector(1,nuse);
	x[iest]=xest;
	if (iest == 1)
		for (j=1;j<=nv;j++) {
			yz[j]=yest[j];
			d[j][1]=yest[j];
			dy[j]=yest[j];
		}
	else {
		m1=(iest < nuse ? iest : nuse);
		for (k=1;k<=m1-1;k++)
			fx[k+1]=x[iest-k]/xest;
		for (j=1;j<=nv;j++) {
			yy=yest[j];
			v=d[j][1];
			c=yy;
			d[j][1]=yy;
			for (k=2;k<=m1;k++) {
				b1=fx[k]*v;
				b=b1-c;
				if (b) {
					b=(c-v)/b;
					ddy=c*b;
					c=b1*b;
				} else
					ddy=v;
				if (k != m1) v=d[j][k];
				d[j][k]=ddy;
				yy += ddy;
			}
			dy[j]=ddy;
			yz[j]=yy;
		}
	}
	free_vector(fx,1,nuse);
}
