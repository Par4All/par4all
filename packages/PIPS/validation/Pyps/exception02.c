int chapi() {
    int a[]={1};
    a[0]+=0;
    return a[0];
}
int chapo() {
    return 1;
}

void patapo() {
}

int main() {
    patapo();
    return chapi() + chapo() ;
}
