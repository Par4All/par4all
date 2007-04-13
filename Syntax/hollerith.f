      program hollerith
     
! this is not an hollerith;-) 
      real*8 hollerith
      
      print *, '"2hab"', 2h'", "'2h  '"

      print *, 10h 2 45 78 0
      print *, 10h1 3456789
      print *, 10h12 45678
      print *, 10h123 567
      print *, 10h1234 6
      print *, 10h 2 45
      print *, 10h1  4
      print *, 10h  3
      print *, 10h 2
      print *, 10h1
      print *, 10h

      print *, 3hFAB, 4hbien, 2h  

      print *, 10h'    abcde

      print *, 10h'1'2' '4'5

      print *, 'foo',  1ha, 1h , 3hx x
      
      a2h=b

      print *, 'foo
     x a la ligne ', 3houi

      print *, 'foo

     x fin ', 3houi

      print *, 'foo
! comments
     x fin 2-', 3houi

      print *, 'foo
      
     x fin 3-', 3houi

      print *, 'foo
                      
     x fin 4-', 3houi

      print *, 3hFAB, "-", 4hbien

      print *, 26habcdefghij          uvwxyz                            OUT

      print *, "2ha b", 2h  , '2ha b', 3 ha b

      print *, 54h0''''5''''0''''5''''0''''5''''0''''5''''0''''5''''0'''

! worst case dilatation
      print *,
     x63h'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

      print *, "padding", 10h
     x , 'ten blanks!'

! bang comments...

      print *, '! pas un commentaire !' ! mais la oui !

      print *, "! et avec ca aussi !" ! hop !

      print *, 10h!123456789! c est tout !

      end
