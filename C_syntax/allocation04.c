/*Memory Allocation test for 4 kinds of variables : local dynamic, local static ,
global and static variables */

int a;//comment
static int m;

/*comment2*/
int foo(int i)
{
	int k = a;
	static int l;
	return k + m;
}
int main()
{	
	int i ;
	int fooi;
	float array[20];
	static int j;
	fooi=foo(i);
	return 0;
}
