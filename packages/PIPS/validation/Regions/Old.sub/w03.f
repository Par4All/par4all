      program w03

      integer i, n
      parameter (n=10)
      real a(n)

      i = 1

      do while (i.le.10)
         a(i) = i*1.23
         i = i + 1
      enddo

      i = 2

      do while (i.lt.n)
         print *, a(i-1)
         i = i + 1
      enddo

      print *, 'bye'

      end
