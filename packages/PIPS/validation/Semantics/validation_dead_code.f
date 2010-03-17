      program validation_dead_code
      integer a, b
      
      j = 3
      
c     if inutile qui doit disparai^tre No 1 (vide de`s le source) :
      if (b .ge. a) then
      else
      endif
      
c     if inutile qui doit disparai^tre No 2 (devient vide par e'limination ) :
      if (b .ge. a) then
c     if faux :
         if (1 .eq. 0) then
            i = 1
         endif
      else
c     if faux :
         if (j .eq. 5) then
            i = 1
         endif
         
      endif

c     Teste les optimisations imbrique'es
c     Doit e^tre re'e'crit :
c     i = -9 (devrait i = 1 et i = -9)
      if (j .ne. 5) then
         do i=1,0
            rien = j+0
         enddo
         if(j.eq.6) then
            i = -8
         else
            i = -9
         endif
      else
         j = 6
      endif

      do i=1,0
         j = j+0
      enddo
c     Bizarre : la pre'condition sur j est ici impre'cise !
c     Devrait avoir i = -2 car j = 3...
      if(j.eq.6) then
         i = -1
      else
         i = -2
      endif
      
c     doit e^tre re'e'crit :
c     j = j + 2
      if (a.gt.10 .and. a.lt.10) then
         j=j+1 
c Si je veux compliquer :-)
c         goto 18
      else
         j=j+2
      endif
      
      if (a.gt.3.and.b.eq.5) then
         rien = 2
      else
         rien = 3
      endif
      print *,j

c     devrait e^tre re'e'crit :
c     rien = rien + 1
c     i = 1
      do 40 i = 0, 1
         rien = rien + 1
 40   continue
      
c     devrait e^tre supprime' :
      do 20 j = i, 0, 2
         rien = rien + 1
 20   continue

 18   i = 2

c     doit e^tre re'e'crit :
c19   i = j + 1
 19   do 30 i = j+1,j,5
c     commentaire sur un goto :
         goto 10
c     supprime' par le controllizer :
         rien = rien +1    
 10      continue
 30   continue
      
c     doit e^tre re'e'crit :
c     j = 1
      if (i .eq. j+1) then
         j = 1
      else
         j = 2
      endif
      print *, j
      
c     doit e^tre supprime' :
      do j = 3,2
         rien = rien +1
         i = 3
      enddo
      
c     doit e^tre supprime' :
      do j = 2, 7, -5
         rien = rien +1
         i = 7
      enddo
      
c     inchange' :
      i = 4
c     doit e^tre re'e'crit :
c     j = 4
      if(i.eq.5) then
         j = 3
      else
         j = 4
      endif

c     doit e^tre re'e'crit :
c     j = 5
      if(.not.i.eq.5) then
         j = 5
      else
         j = 6
      endif

      if (i .eq. 5) goto 31
      print *, 'salut'
 31   a = 9
      
c     Ce test est en soit infaisable :
c     doit e^tre re'e'crit :
c     j = 8
      if(i.gt.5.and.3.gt.i) then
         j = 7
      else
         j = 8
      endif

c     de'pendance sur ce qui pre'ce`de :
c     doit e^tre re'e'crit :
c     k = 1
c     goto 100
      if(j.ge.3) then
         k = 3
         goto 100
      else
         k = 1
         goto 100
      endif
      print *, j
      
      goto 100

c     supprime' par le controllizer :
      j = -i

      stop
c     Quel est donc la pre'condition ici ?
c     supprime' par le controllizer :
      j = i
      
      stop
c     doit e^tre conserve' :
 100  i = i + 1
      
      stop
      
c     doit e^tre supprime' :
      j = a
      
      end
