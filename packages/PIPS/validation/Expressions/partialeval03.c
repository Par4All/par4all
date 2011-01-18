int f(int* i)
{
        return *i+1;
}

int main()
{
        int i;
        i = 5;
        f(&i);
        return i;
}

