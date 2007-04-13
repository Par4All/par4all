      program w09

C     Test de l'arret de la voiture propose par Nicolas Halbwachs, 15
C     mars 2005

      integer ms, s, m

      ms = 0
      s = 0
      m = 0

      do while(ms.le.2.and.s.le.4)
         if(x.gt.0.) then
            s = s + 1
            ms = 0
         else
            m = m + 1
            ms = ms + 1
         endif
      enddo

      print *, ms, s, m
      read *, s, ms

      if(m.le.11) then
         print *, "healthy"
      else
         print *, "crashed!"
      endif

      end
