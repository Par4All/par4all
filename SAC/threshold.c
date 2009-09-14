void threshold(int data[128],int val)
{
    int i;
    for(i=0;i<128;++i)
        if(data[i]>val)
            data[i]=val;
}

void main()
{
    int data[128],i;
    threshold(data,10);
    for(i=0;i<128;i++)
        printf("%d",i);
}
