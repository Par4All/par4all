#include <math.h>

void tptest(data1,data2,n,t,prob)
float data1[],data2[],*t,*prob;
int n;
{
	int j;
	float var1,var2,ave1,ave2,sd,df,cov=0.0;
	void avevar();
	float betai();

	avevar(data1,n,&ave1,&var1);
	avevar(data2,n,&ave2,&var2);
	for (j=1;j<=n;j++)
		cov += (data1[j]-ave1)*(data2[j]-ave2);
	cov /= df=n-1;
	sd=sqrt((var1+var2-2.0*cov)/n);
	*t=(ave1-ave2)/sd;
	*prob=betai(0.5*df,0.5,df/(df+(*t)*(*t)));
}
