#include <math.h>

void ttest(data1,n1,data2,n2,t,prob)
int n1,n2;
float data1[],data2[],*t,*prob;
{
	float var1,var2,svar,df,ave1,ave2;
	void avevar();
	float betai();

	avevar(data1,n1,&ave1,&var1);
	avevar(data2,n2,&ave2,&var2);
	df=n1+n2-2;
	svar=((n1-1)*var1+(n2-1)*var2)/df;
	*t=(ave1-ave2)/sqrt(svar*(1.0/n1+1.0/n2));
	*prob=betai(0.5*df,0.5,df/(df+(*t)*(*t)));
}
