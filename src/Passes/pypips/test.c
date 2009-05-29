int foo(int a) {
    return a;
}

void bar(int *c) {
    *c=foo(2);
}
void malabar(int *c) {
    *c=foo(3);
}
