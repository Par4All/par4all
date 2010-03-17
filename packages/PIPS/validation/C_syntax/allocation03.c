//Check Memory Allocation offsets for local dynamic and static variables of known sizes in one C file
int a;//comment
/*comment2*/
void foo()
{
	int k;
	static int l;
	//return k;
}
int main()
{	
	int i ;
	int fooi;
	float array[20];
	static int j;
	foo();
	return 0;
}
