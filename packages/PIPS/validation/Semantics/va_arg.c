int f(int a, ...)
{
	return a;
}

int main()
{
	int a,b,c;
	// This works 
	// f(a,b,c);
	// but not this...
	f(a,b);
	return 0;
}
