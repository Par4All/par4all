#include <math.h>

static float sqrarg;
#define SQR(a) (sqrarg=(a),sqrarg*sqrarg)

void spear(data1,data2,n,d,zd,probd,rs,probrs)
float data1[],data2[];
float *d,*zd,*probd,*rs,*probrs;
int n;
{
	int j;
	float vard,t,sg,sf,fac,en3n,en,df,aved,*wksp1,*wksp2;
	void sort2(),crank(),free_vector();
	float erfcc(),betai(),*vector();

	wksp1=vector(1,n);
	wksp2=vector(1,n);
	for (j=1;j<=n;j++) {
		wksp1[j]=data1[j];
		wksp2[j]=data2[j];
	}
	sort2(n,wksp1,wksp2);
	crank(n,wksp1,&sf);
	sort2(n,wksp2,wksp1);
	crank(n,wksp2,&sg);
      	*d=0.0;
	for (j=1;j<=n;j++)
		*d += SQR(wksp1[j]-wksp2[j]);
	en=n;
	en3n=en*en*en-en;
	aved=en3n/6.0-(sf+sg)/12.0;
	fac=(1.0-sf/en3n)*(1.0-sg/en3n);
	vard=((en-1.0)*en*en*SQR(en+1.0)/36.0)*fac;
	*zd=(*d-aved)/sqrt(vard);
	*probd=erfcc(fabs(*zd)/1.4142136);
	*rs=(1.0-(6.0/en3n)*(*d+0.5*(sf+sg)))/fac;
	t=(*rs)*sqrt((en-2.0)/((*rs+1.0)*(1.0-(*rs))));
	df=en-2.0;
	*probrs=betai(0.5*df,0.5,df/(df+t*t));
	free_vector(wksp2,1,n);
	free_vector(wksp1,1,n);
}

#undef SQR
