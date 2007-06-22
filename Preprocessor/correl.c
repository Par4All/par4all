void correl(data1,data2,n,ans)
float data1[],data2[],ans[];
int n;
{
	int no2,i;
	float dum,*fft,*vector();
	void twofft(),realft(),free_vector();

	fft=vector(1,2*n);
	twofft(data1,data2,fft,ans,n);
	no2=n/2;
	for (i=2;i<=n+2;i+=2) {
		ans[i-1]=(fft[i-1]*(dum=ans[i-1])+fft[i]*ans[i])/no2;
		ans[i]=(fft[i]*dum-fft[i-1]*ans[i])/no2;
	}
	ans[2]=ans[n+1];
	realft(ans,no2,-1);
	free_vector(fft,1,2*n);
}
