void red(iz1,iz2,jz1,jz2,jm1,jm2,jmf,ic1,jc1,jcf,kc,c,s)
float ***c,**s;
int iz1,iz2,jz1,jz2,jm1,jm2,jmf,ic1,jc1,jcf,kc;
{
	int loff,l,j,ic,i;
	float vx;

	loff=jc1-jm1;
	ic=ic1;
	for (j=jz1;j<=jz2;j++) {
		for (l=jm1;l<=jm2;l++) {
			vx=c[ic][l+loff][kc];
			for (i=iz1;i<=iz2;i++) s[i][l] -= s[i][j]*vx;
		}
		vx=c[ic][jcf][kc];
		for (i=iz1;i<=iz2;i++) s[i][jmf] -= s[i][j]*vx;
		ic += 1;
	}
}
