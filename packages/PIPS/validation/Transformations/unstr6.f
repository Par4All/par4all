      program unstr6
c     This program should be reduced to the empty program
      integer a, b
      
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
