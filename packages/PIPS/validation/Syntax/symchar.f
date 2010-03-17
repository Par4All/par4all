      program symchar

C     Bug trouve dans Scilab, matops.f

      integer bufsize
      parameter (bufsize=4096)
      character buffer*(bufsize)

      buffer = 'Hello world!'

      print *, buffer

      end
