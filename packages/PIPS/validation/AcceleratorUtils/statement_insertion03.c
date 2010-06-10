void pied(int a[10])
{
    int i;
    for(i=0;i<10;i++) a[i]=0;
#pragma pips inserted statement to check
    for(i=10;i<11;i++) a[i]=0;
}
int main(int argc, char *argv[])
{
    int i,a[10];
    pied(a);
    return 0;
}
