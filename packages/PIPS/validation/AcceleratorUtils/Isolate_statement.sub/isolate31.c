typedef struct { char data[300] ; } type;

int pain( type rdata[10]) {
    int i;
holy:for(i=0;i<100;i++)
        rdata[3].data[i]= 'e';
}
int main() {
    type R[10];
    pain(R);
    return R[3].data[20]-'e';
}
