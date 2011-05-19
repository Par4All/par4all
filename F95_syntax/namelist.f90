      PROGRAM test
      integer n
      real  x
      namelist / struct / n,x
      read (*, nml=struct)
      x=x+1
      n=n+1
      write (*, nml=struct)
      return
      end
