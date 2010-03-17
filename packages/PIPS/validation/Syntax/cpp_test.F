! test le preprocessing cpp de pips
      program cpp

#include "cpp_test.h"

#ifdef __PIPS__
      print *, '__PIPS__ est defini'
#else
      print *, '__PIPS__ n''est pas defini'
#endif

#ifdef __HPFC__
      print *, '__HPFC__ est defini'
#else
      print *, '__HPFC__ n''est pas defini'
#endif

      end
