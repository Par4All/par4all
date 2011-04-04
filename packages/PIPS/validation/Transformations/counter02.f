      program counter02
      integer n, i
      read *, n
      if (n.lt.100) then
        print *, 'n < 100'
      else
        print *, 'n >= 100'
      endif
      do i=1, n
        print *, i
      enddo
      print *, 'done'
      end
