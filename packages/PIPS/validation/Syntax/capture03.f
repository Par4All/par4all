C     test de la (non) capture des intrinsics, e.g. MIN used as a 
C     statement function, while another user module uses a function
C     with the same name

      program capture03
C     Here "min" is the Fortran intrinsics
      i = min(2, 3)
      print *, i
      call foo(i)
      print *, i
      end

      subroutine foo(k)
c     Here, "min" is a user function
      external min
      k = min(k,k)
      end

      function min(l,m)
      min = l + m
      end
