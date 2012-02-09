typedef struct { char data[300] ; } type;
int pain( type* rdata) {
    int i;
holy:for(i=0;i<100;i++)
        rdata[0].data[i]= 'e';
}
int main() {
    type R;
    pain(&R);
    return R.data[20]-'e';
}
