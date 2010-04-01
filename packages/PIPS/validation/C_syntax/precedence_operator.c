/* Check that the precedence based parentheses elimination algorithm
    works for casts */

struct test2
{
    int a;
};

struct test1
{
    struct test2 t2;
};

int main()
{
    struct test1 t1;
    void* t1_ptr=&t1;

    int i=((struct test1*)t1_ptr)->t2.a;

    return 0;
}
