int f(int a, ...)
{
	return a;
}

int main()
{
	int a = 1, b, c;
	// This works
	// f(a,b,c);
	// but not this...
	c = f(a,b);
	return c;
}
