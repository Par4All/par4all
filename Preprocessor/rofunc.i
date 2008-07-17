#include <math.h>

extern int ndatat;	/* defined in MEDFIT */
extern float *xt,*yt,aa,abdevt;

float rofunc(b)
float b;
{
	int j,n1,nmh,nml;
	float *arr,d,sum=0.0,*vector();
	void sort(),free_vector();

	arr=vector(1,ndatat);
	n1=ndatat+1;
	nml=n1/2;
	nmh=n1-nml;
	for (j=1;j<=ndatat;j++) arr[j]=yt[j]-b*xt[j];
	sort(ndatat,arr);
	aa=0.5*(arr[nml]+arr[nmh]);
	abdevt=0.0;
	for (j=1;j<=ndatat;j++) {
		d=yt[j]-(b*xt[j]+aa);
		abdevt += fabs(d);
		sum += d > 0.0 ? xt[j] : -xt[j];
	}
	free_vector(arr,1,ndatat);
	return sum;
}
