	program bar
        real t(100,10)
	x = 2
        do 20 i=1,10
           do 20 j=1,10,2
              t(i,j) = t(i,j)+1
 20        continue
	end
