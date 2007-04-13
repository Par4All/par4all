      subroutine equiv19(n)

C     Goal: check that an equivalence between static variables
C     is properly handled.

      data k /3/
      save m
      equivalence (m,k)

      print *, n

      end
