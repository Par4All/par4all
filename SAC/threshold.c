void threshold(float data[128],float val)
{
    int i;
    for(i=0;i<128;++i)
        if(data[i]>val)
            data[i]=val;
}

void main()
{
    float data[128],i;
    threshold(data,10.);
    for(i=0;i<128;i++)
        printf("%d",i);
}
