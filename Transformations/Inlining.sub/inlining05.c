static int pmax(int a, int b)
{
    int argc =a > b ? a : b;
    return argc;
}

int main(int argc, char **argv)
{
    int c ;
    c = pmax(2,argc);
    return 0;
}

