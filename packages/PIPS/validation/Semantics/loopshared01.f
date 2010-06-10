      program loopshared01

C     Check conformance with Fortran standard, section 11.10, pp. 11-5,
C     11-9

      n = 0
      do 100 i = 1, 10
         j = i
         do 100 k = 1, 5
            l = k
 100        n = n + 1

!     n == 50
      print *, i, j, k, l, n

      n = 0
      do 200 i = 1, 10
         j = i
         do 200 k = 5, 1
            l = k
 200        n = n + 1

!     n == 0
      print *, i, j, k, l, n

      end
