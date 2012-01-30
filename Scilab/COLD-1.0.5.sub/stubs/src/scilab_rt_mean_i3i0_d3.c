
void scilab_rt_mean_i3i0_d3(int sin00,int sin01, int sin02, double in0[sin00][sin01][sin02],
    double in1,
    int sout00, int sout01, int sout02, double out0[sout00][sout01][sout02])
{

  int lv1, lv2, lv3;
  double val0;

  if (in1) {

    for (lv1=0; lv1<sin00; ++lv1){
      for (lv2=0; lv2<sin01; ++lv2){
        for (lv3=0; lv2<sin02; ++lv3){
          val0 += in0[lv1][lv2][lv3];
        }
      }
    }
    
    for (lv1=0; lv1<sout00; ++lv1){
      for (lv2=0; lv2<sout01; ++lv2){
        for (lv3=0; lv3<sout02; ++lv3){
          out0[lv1][lv2][lv3] = val0;
        }
      }
    }
  }

}
