void cmp(float data[128],float val)
{
    int i,j;
    for(i=0;i<128;++i)
        j=data[i]>val;
}

int main()
{
    float data[128];
    int i;
    for(i=0;i<128;i++)
        data[i]=i;
    cmp(data,10.f);
    for(i=0;i<128;i++)
        printf("%d",i);
    return 0;
}
