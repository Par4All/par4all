void locate(xx,n,x,j)
float xx[],x;
int n,*j;
{
	int ascnd,ju,jm,jl;

	jl=0;
	ju=n+1;
	ascnd=xx[n] > xx[1];
	while (ju-jl > 1) {
		jm=(ju+jl) >> 1;
		if (x > xx[jm] == ascnd)
			jl=jm;
		else
			ju=jm;
	}
	*j=jl;
}
