# -*- coding: utf-8 -*-

import os, sys, Image


class Images(object):

    size = (512, 512)

    def create_thumbnail(self, image):
        outfile = os.path.splitext(image)[0] + '.thumbnail.png'
	try:
            ii = Image.open(image)
            ii.thumbnail(self.size)
            ii.save(outfile, 'PNG')
	except IOError:
            print 'ERROR'
        return outfile[len(paws.public) - 1 : ]
 
    def create_zoom_image(self, image):
        outfile = self.create_thumbnail(paws.results + image)
	print 'o', outfile
	return '<a href="/' + paws.result_dir + image + '" class=' + paws.images_class + ' ><img src="' + outfile + '" /></a>'
	
