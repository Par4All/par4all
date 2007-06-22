      program hierarchize1

      print *,1

      if (a.eq.0) goto 9

      print *,2
      
 3    print *,3
      
      if (b.eq.0) goto 5
      
 4    print *,4
      goto 3
      
 5    print *,5
      if (c.eq.0) goto 7

 6    print *,6
      goto 4

 7    print *,7     
      
 9    print *,9

      end
