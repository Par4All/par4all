
void scilab_rt_datevec_d2_i2i2i2i2i2d2(int sin00, int sin01, double in0[sin00][sin01], 
    int sout00, int sout01, int Year[sout00][sout01],
    int sout10, int sout11, int Month[sout10][sout11],
    int sout20, int sout21, int Day[sout20][sout21],
    int sout30, int sout31, int Hour[sout30][sout31],
    int sout40, int sout41, int Min[sout40][sout41],
    int sout50, int sout51, double Sec[sout50][sout51])
{
  int i, j;
  double val0;

  for (i = 0; i < sin00; ++i) {
    for (j = 0; j < sin01; ++j) {
      val0 += in0[i][j];
    }
  }
  
  for (i = 0; i < sout00; ++i) {
    for (j = 0; j < sout01; ++j) {
      Year[i][j] = (int) val0;
    }
  }
  for (i = 0; i < sout10; ++i) {
    for (j = 0; j < sout11; ++j) {
      Month[i][j] = (int) val0;
    }
  }
  for (i = 0; i < sout20; ++i) {
    for (j = 0; j < sout21; ++j) {
      Day[i][j] = (int) val0;
    }
  }
  for (i = 0; i < sout30; ++i) {
    for (j = 0; j < sout31; ++j) {
      Hour[i][j] = (int) val0;
    }
  }
  for (i = 0; i < sout40; ++i) {
    for (j = 0; j < sout41; ++j) {
      Min[i][j] = (int) val0;
    }
  }
  for (i = 0; i < sout50; ++i) {
    for (j = 0; j < sout51; ++j) {
      Sec[i][j] = val0;
    }
  }


}


