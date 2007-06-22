void mdian1(x,n,xmed)
float x[],*xmed;
int n;
{
	int n2,n2p;
	void sort();

	sort(n,x);
	n2p=(n2=n/2)+1;
	*xmed=(n % 2 ? x[n2p] : 0.5*(x[n2]+x[n2p]));
}
