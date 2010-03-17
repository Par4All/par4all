#include <math.h>

static float sqrarg;
#define SQR(a) (sqrarg=(a),sqrarg*sqrarg)

void memcof(data,n,m,pm,cof)
float data[],*pm,cof[];
int n,m;
{
	int k,j,i;
	float p=0.0,*wk1,*wk2,*wkm,*vector();
	void free_vector();

	wk1=vector(1,n);
	wk2=vector(1,n);
	wkm=vector(1,m);
	for (j=1;j<=n;j++) p += SQR(data[j]);
	*pm=p/n;
	wk1[1]=data[1];
	wk2[n-1]=data[n];
	for (j=2;j<=n-1;j++) {
		wk1[j]=data[j];
		wk2[j-1]=data[j];
	}
	for (k=1;k<=m;k++) {
		float num=0.0,denom=0.0;
		for (j=1;j<=(n-k);j++) {
			num += wk1[j]*wk2[j];
			denom += SQR(wk1[j])+SQR(wk2[j]);
		}
		cof[k]=2.0*num/denom;
		*pm *= (1.0-SQR(cof[k]));
		if (k != 1)
			for (i=1;i<=(k-1);i++)
				cof[i]=wkm[i]-cof[k]*wkm[k-i];
		if (k == m) {
			free_vector(wkm,1,m);
			free_vector(wk2,1,n);
			free_vector(wk1,1,n);
			return;
		}
		for (i=1;i<=k;i++) wkm[i]=cof[i];
		for (j=1;j<=(n-k-1);j++) {
			wk1[j] -= wkm[k]*wk2[j];
			wk2[j]=wk2[j+1]-wkm[k]*wk1[j+1];
		}
	}
}

#undef SQR
