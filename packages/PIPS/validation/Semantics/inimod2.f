      subroutine inimod2

      parameter  ( mp = 402, np = 160 )
      real ro(mp,np)

      do 5 j=1,np
         do 5 i=1,mp
            read(11,1000,end=99) ro(i,j)
 5    continue

      print *, i
 99   stop
      end
