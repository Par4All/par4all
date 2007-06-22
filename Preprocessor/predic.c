void predic(data,ndata,d,npoles,future,nfut)
float data[],d[],future[];
int ndata,npoles,nfut;
{
	int k,j;
	float sum,discrp,*reg,*vector();
	void free_vector();

	reg=vector(1,npoles);
	for (j=1;j<=npoles;j++) reg[j]=data[ndata+1-j];
	for (j=1;j<=nfut;j++) {
		discrp=0.0;
		sum=discrp;
		for (k=1;k<=npoles;k++) sum += d[k]*reg[k];
		for (k=npoles;k>=2;k--) reg[k]=reg[k-1];
		future[j]=reg[1]=sum;
	}
	free_vector(reg,1,npoles);
}
