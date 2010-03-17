int volume(int a[256][256])
{
    int v=0;
    int i,j;
volume:    for(i=0;i<256;i++)
           {
               v=v+1;
               for(j=0;j<256;j++)
                   v=v+a[i][j];
           }
    return v;
}

main()
{
    int a [256][256];
    int n = volume(a);
    printf("%d",n);
    return 0;
}
