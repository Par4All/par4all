

void scilab_rt_mean_d3s0_d3(int si00,int si01, int si02, double in0[si00][si01][si02],
    char* in1,
    int so00, int so01, int so02, double out0[so00][so01][so02])
{

  int lv1, lv2, lv3;
  double val0;

  if (in1) {

    for (lv1=0; lv1<si00; ++lv1){
      for (lv2=0; lv2<si01; ++lv2){
        for (lv3=0; lv2<si02; ++lv3){
          val0 += in0[lv1][lv2][lv3];
        }
      }
    }
    
    for (lv1=0; lv1<so00; ++lv1){
      for (lv2=0; lv2<so01; ++lv2){
        for (lv3=0; lv3<so02; ++lv3){
          out0[lv1][lv2][lv3] = val0;
        }
      }
    }
  }

}
