enum{N=100};

void P4A_scmp_kernel_0(int N, int a[N])
{
   int i;
   for(i = 0; i <= 99; i += 1)
      a[i] += 3;

}

int main()
{

   int a[N] = {0};
   
   
 kernel: P4A_scmp_kernel_0(N, a);
   
   return 0;
}
