#include <math.h>

static float sqrarg;
#define SQR(a) (sqrarg=(a),sqrarg*sqrarg)

void tutest(data1,n1,data2,n2,t,prob)
float data1[],data2[],*t,*prob;
int n1,n2;
{
	float var1,var2,df,ave1,ave2;
	void avevar();
	float betai();

	avevar(data1,n1,&ave1,&var1);
	avevar(data2,n2,&ave2,&var2);
	*t=(ave1-ave2)/sqrt(var1/n1+var2/n2);
	df=SQR(var1/n1+var2/n2)/(SQR(var1/n1)/(n1-1)+SQR(var2/n2)/(n2-1));
	*prob=betai(0.5*df,0.5,df/(df+SQR(*t)));
}

#undef SQR
