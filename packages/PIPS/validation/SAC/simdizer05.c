main()
{
    float a[4]={0,1,2,3};
    float b[4]={3,2,1,0};

    /* this should be a first pack */
    a[0]=b[0]+a[0];
    a[1]=a[1]+b[1];
    a[2]=b[2]+a[2];
    a[3]=b[3]+a[3];
}
