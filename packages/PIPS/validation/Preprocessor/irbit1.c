int irbit1(iseed)
unsigned long *iseed;
{
        unsigned long newbit;

        newbit = (*iseed & 131072) >> 17
                ^ (*iseed & 16) >> 4
                ^ (*iseed & 2) >> 1
                ^ (*iseed & 1);
        *iseed=(*iseed << 1) | newbit;
        return (int) newbit;
}
