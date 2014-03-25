int duff_device01(char *from, char *to, int count)
{
    int i = 0;

 loop1:
    for(i = 0; i < count; i++) {
        to[i] = from[i];
    }

    return i;
}
