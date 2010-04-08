main(int argc, char *argv[])
{
    struct _ { int *a ; int c[4]; } __;
    int b = atoi(argv[1]);
    do {
        __.a = malloc(sizeof(int)*4);
        __.a[0]=b*2;
        __.a[1]=b*3;
        __.a[2]=b*4;
        __.a[3]=b*5;
    } while(0);
    do {
        __.c[0]=b*2;
        __.c[1]=b*3;
        __.c[2]=b*4;
        __.c[3]=b*5;
    } while(0);
    return 0;
}
