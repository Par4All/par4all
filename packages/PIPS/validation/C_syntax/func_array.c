int add(int a,int b){return a+b;}

typedef int (*FUNC)(int a,int b);

static FUNC functions[1] = { add };

int main() 
{
    FUNC fu=*(functions[0]);
    return (*fu)(1,1);
}

