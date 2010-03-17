void avevar(data,n,ave,svar)
float data[],*ave,*svar;
int n;
{
	int j;
	float s;

	*ave=(*svar)=0.0;
	for (j=1;j<=n;j++) *ave += data[j];
	*ave /= n;
	for (j=1;j<=n;j++) {
		s=data[j]-(*ave);
		*svar += s*s;
	}
	*svar /= (n-1);
}
