int main()
{
    int i=0;
    if(i)
    {
        label:
        {
            i=1;
        }
    }
    if(!i)
        goto label;
    return 0;
}
