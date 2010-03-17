#include <stdio.h>
#include <math.h>

#define TFACTR 0.9
#define ALEN(a,b,c,d) sqrt(((b)-(a))*((b)-(a))+((d)-(c))*((d)-(c)))

void anneal(x,y,iorder,ncity)
float x[],y[];
int iorder[],ncity;
{
	int ans,nover,nlimit,i1,i2,idum;
	unsigned long int iseed;
	int i,j,k,nsucc,nn,idec;
	static int n[7];
	float path,de,t;
	float ran3();
	int irbit1(),metrop();
	void reverse(),trnspt();
	float revcst(),trncst();

	nover=100*ncity;
	nlimit=10*ncity;
	path=0.0;
	t=0.5;
	for (i=1;i<ncity;i++) {
		i1=iorder[i];
		i2=iorder[i+1];
		path += ALEN(x[i1],x[i2],y[i1],y[i2]);
	}
	i1=iorder[ncity];
	i2=iorder[1];
	path += ALEN(x[i1],x[i2],y[i1],y[i2]);
	idum = -1;
	iseed=111;
	for (j=1;j<=100;j++) {
		nsucc=0;
		for (k=1;k<=nover;k++) {
			do {
				n[1]=1+(int) (ncity*ran3(&idum));
				n[2]=1+(int) ((ncity-1)*ran3(&idum));
				if (n[2] >= n[1]) ++n[2];
				nn=1+((n[1]-n[2]+ncity-1) % ncity);
			} while (nn<3);
			idec=irbit1(&iseed);
			if (idec == 0) {
				n[3]=n[2]+(int) (abs(nn-2)*ran3(&idum))+1;
				n[3]=1+((n[3]-1) % ncity);
				de=trncst(x,y,iorder,ncity,n);
				ans=metrop(de,t);
				if (ans) {
					++nsucc;
					path += de;
					trnspt(iorder,ncity,n);
				}
			} else {
				de=revcst(x,y,iorder,ncity,n);
				ans=metrop(de,t);
				if (ans) {
					++nsucc;
					path += de;
					reverse(iorder,ncity,n);
				}
			}
			if (nsucc >= nlimit) break;
		}
		printf("\n %s %10.6f %s %12.6f \n","T =",t,
			"	 Path Length =",path);
		printf("Successful Moves: %6d\n",nsucc);
		t *= TFACTR;
		if (nsucc == 0) return;
	}
}

float revcst(x,y,iorder,ncity,n)
float x[],y[];
int iorder[],ncity,n[];
{
	float xx[5],yy[5],de;
	int j,ii;

	n[3]=1 + ((n[1]+ncity-2) % ncity);
	n[4]=1 + (n[2] % ncity);
	for (j=1;j<=4;j++) {
		ii=iorder[n[j]];
		xx[j]=x[ii];
		yy[j]=y[ii];
	}
	de = -ALEN(xx[1],xx[3],yy[1],yy[3]);
	de -= ALEN(xx[2],xx[4],yy[2],yy[4]);
	de += ALEN(xx[1],xx[4],yy[1],yy[4]);
	de += ALEN(xx[2],xx[3],yy[2],yy[3]);
	return de;
}

void reverse(iorder,ncity,n)
int iorder[],ncity,n[];
{
	int nn,j,k,l,itmp;

	nn=(1+((n[2]-n[1]+ncity) % ncity))/2;
	for (j=1;j<=nn;j++) {
		k=1 + ((n[1]+j-2) % ncity);
		l=1 + ((n[2]-j+ncity) % ncity);
		itmp=iorder[k];
		iorder[k]=iorder[l];
		iorder[l]=itmp;
	}
}

float trncst(x,y,iorder,ncity,n)
float x[],y[];
int iorder[],ncity,n[];
{
	float xx[7],yy[7],de;
	int j,ii;

	n[4]=1 + (n[3] % ncity);
	n[5]=1 + ((n[1]+ncity-2) % ncity);
	n[6]=1 + (n[2] % ncity);
	for (j=1;j<=6;j++) {
		ii=iorder[n[j]];
		xx[j]=x[ii];
		yy[j]=y[ii];
	}
	de = -ALEN(xx[2],xx[6],yy[2],yy[6]);
	de -= ALEN(xx[1],xx[5],yy[1],yy[5]);
	de -= ALEN(xx[3],xx[4],yy[3],yy[4]);
	de += ALEN(xx[1],xx[3],yy[1],yy[3]);
	de += ALEN(xx[2],xx[4],yy[2],yy[4]);
	de += ALEN(xx[5],xx[6],yy[5],yy[6]);
	return de;
}

void trnspt(iorder,ncity,n)
int iorder[],ncity,n[];
{
	int m1,m2,m3,nn,j,jj,*jorder,*ivector();
	void free_ivector();

	jorder=ivector(1,ncity);
	m1=1 + ((n[2]-n[1]+ncity) % ncity);
	m2=1 + ((n[5]-n[4]+ncity) % ncity);
	m3=1 + ((n[3]-n[6]+ncity) % ncity);
	nn=1;
	for (j=1;j<=m1;j++) {
		jj=1 + ((j+n[1]-2) % ncity);
		jorder[nn++]=iorder[jj];
	}
	if (m2>0) {
		for (j=1;j<=m2;j++) {
			jj=1+((j+n[4]-2) % ncity);
			jorder[nn++]=iorder[jj];
		}
	}
	if (m3>0) {
		for (j=1;j<=m3;j++) {
			jj=1 + ((j+n[6]-2) % ncity);
			jorder[nn++]=iorder[jj];
		}
	}
	for (j=1;j<=ncity;j++)
		iorder[j]=jorder[j];
	free_ivector(jorder,1,ncity);
}

int metrop(de,t)
float de,t;
{
	static int gljdum=1;
	float ran3();

	return de < 0.0 || ran3(&gljdum) < exp(-de/t);
}

#undef TFACTR
#undef ALEN
