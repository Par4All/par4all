	integer function minimu(i,j)

	if(i.gt.j) then
		minimu = j
	else
		minimu = i
	endif
	print *, i, j, minimu
	return
	end
