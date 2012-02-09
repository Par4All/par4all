int *global;
void bra(int y[3]) { y[1]=0;}
void pain() {
    global=malloc(sizeof(int)*3);
holy: bra(global);
}
int main() {
    pain();
    return global[1];
}

