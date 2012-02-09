int pain( int rdata[100]) {
    int i;
holy:for(i=0;i<100;i++)
        rdata[i]=rdata[0];
}
int main() {
    int R[100];
    R[0]=0;
    pain(R);
    return R[30];
}
