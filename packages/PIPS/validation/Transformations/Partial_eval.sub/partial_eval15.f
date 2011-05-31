C Checking results for Fortran integer division as opposed to C integer division

      program main
      logical check_mod_div

      i = 3/ 2
      print *, i, "must be 1 and check_mod_div(3,2)",
     &     check_mod_div(3,2)

      i = (-3)/ 2
      print *, i, "must be -1 and check_mod_div(-3,2)",
     &     check_mod_div(-3,2)

      i = 3/(-2)
      print *, i, "must be -1 and check_mod_div(3,-2)",
     &     check_mod_div(3,-2)

      i = (-3)/(-2)
      print *, i, "must be 1 and check_mod_div(-3,-2)",
     &     check_mod_div(-3,-2)

      end

      logical function check_mod_div(i,j)
      check_mod_div = i.eq.((i/j)*j+mod(i,j))
      end
