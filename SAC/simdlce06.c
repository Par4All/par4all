void memstore(void *dest,void *src,int sz)
{
    my_memcpy(dest,src,sz);
}
void caram()
{
    int j[4],k[4];
    while(1) {
        j[0]=1;
        my_memcpy(&k[0],&j[0],4*sizeof(int));
        memstore(&j[0],&k[0],4*sizeof(int));

    }

}
