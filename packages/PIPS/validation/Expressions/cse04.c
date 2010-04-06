void alphablending(short src0[512][512], short src1[512][512], short result[512][512])
{
    unsigned int i, j;
    //PIPS generated variable
    unsigned int i_t, j_t;
l99998:
    for(i_t = 1; i_t <= 4; i_t += 1)
        for(j_t = 1; j_t <= 64; j_t += 1)
            for(i = 1; i <= 128; i += 1)
                for(j = 1; j <= 8; j += 1)
                    result[i+128*i_t-129][j+8*j_t-9] = (40*src0[i+128*i_t-129][j+8*j_t-9]+60*src1[i+128*i_t-129][j+8*j_t-9])/100;
}
