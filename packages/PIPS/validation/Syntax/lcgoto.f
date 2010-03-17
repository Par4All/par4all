      program lcgoto
c     check labelled computed go to's

      go to 10

 10   go to(100,200) i

 100  print *, 100
 200  print *, 200

      end
