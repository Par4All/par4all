      program unstr
      integer a, b
      
c     10 j = 3
 10   j = 3
      goto 20
c     20   continue: could be fuse with node 30
 20   continue
      continue

c     if (a .eq. 0) goto 30
 30   if (a .eq. 0) goto 30
c     if:
      if (b .eq. 0) goto 20
      goto 10

      end
