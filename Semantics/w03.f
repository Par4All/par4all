      program w03

      integer i, n
      parameter (n=10)
      real a(n)

      i = 0

      do while (.false.)
         i = i + 1
         print *, 'never'
      enddo

      print *, i

      do while (.true.)
         print *, 'always'
      enddo

      print *, 'never again', i

      end
