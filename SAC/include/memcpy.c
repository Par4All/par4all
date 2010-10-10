int my_memcpy(void *dest, void *src,unsigned int nmemb)
{
    char *idest=dest,*isrc=src;
    for(;nmemb>=0;nmemb--)
        idest[nmemb-1]=isrc[nmemb-1];
}
