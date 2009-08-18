void threshold(int data[128],int val)
{
    int i;
    for(i=0;i<128;++i)
        if(data[i]>val)
            data[i]=val;
}
