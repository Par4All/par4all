int pof(){
    int uV[3][30], uR[3][30];
    int lv1, lv0,result1=0,result2=0;
    for(lv0=0;lv0<=2;lv0+=1){
        for(lv1=0;lv1<=29;lv1+=1){
            result1+=uV[lv0][lv1];
            result2+=uR[lv0][lv1];
        }
    }
    return result1+result2;
}
