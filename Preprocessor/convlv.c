static float sqrarg;
#define SQR(a) (sqrarg=(a),sqrarg*sqrarg)

void convlv(data,n,respns,m,isign,ans)
float data[],respns[],ans[];
int n,m,isign;
{
	int i,no2;
	float dum,mag2,*fft,*vector();
	void twofft(),realft(),nrerror(),free_vector();

	fft=vector(1,2*n);
	for (i=1;i<=(m-1)/2;i++)
		respns[n+1-i]=respns[m+1-i];
	for (i=(m+3)/2;i<=n-(m-1)/2;i++)
		respns[i]=0.0;
	twofft(data,respns,fft,ans,n);
	no2=n/2;
	for (i=2;i<=n+2;i+=2) {
		if (isign == 1) {
			ans[i-1]=(fft[i-1]*(dum=ans[i-1])-fft[i]*ans[i])/no2;
			ans[i]=(fft[i]*dum+fft[i-1]*ans[i])/no2;
		} else if (isign == -1) {
 		        if ((mag2=SQR(ans[i-1])+SQR(ans[i])) == 0.0)
				nrerror("Deconvolving at response zero in CONVLV");
			ans[i-1]=(fft[i-1]*(dum=ans[i-1])+fft[i]*ans[i])/mag2/no2;
			ans[i]=(fft[i]*dum-fft[i-1]*ans[i])/mag2/no2;
		} else nrerror("No meaning for ISIGN in CONVLV");
	}
	ans[2]=ans[n+1];
	realft(ans,no2,-1);
	free_vector(fft,1,2*n);
}

#undef SQR
