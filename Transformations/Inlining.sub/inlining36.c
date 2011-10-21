void idhem(int *a) {
    *a=*a;
}
void mehdi(int *b) {
    *b=*b;
    idhem(b);
}

