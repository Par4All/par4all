void memstore(void *dest,void *src,int sz)
{
    my_memcpy(dest,src,sz);
}
void caram()
{
    int i,j[4],k[4];
    for(i=0;i<10;i++) {
        my_memcpy(&k[0],&j[0],4*sizeof(int));
        memstore(&j[0],&k[0],4*sizeof(int));
    }

}
