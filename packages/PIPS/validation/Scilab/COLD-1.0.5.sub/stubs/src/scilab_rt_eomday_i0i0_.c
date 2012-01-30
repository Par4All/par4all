int scilab_rt_eomday_i0i0_(int year, int month){


  int common_year[12] = {31,28,31,30,31,30,31,31,30,31,30,31};
  int leap_year[12]   = {31,29,31,30,31,30,31,31,30,31,30,31};
  int last_day = 0;

  last_day = leap_year[month-1] +  common_year[month-1] + year;
  return last_day;
} 

