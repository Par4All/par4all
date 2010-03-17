      program close56

c     Problems with IO statements

c     This one is not f77 compatible
c     print 56, i
      print *, i

      write (56) i

      close(56)

c     This one should be ok
c      close((56))

      close(unit=i*j)

      end
