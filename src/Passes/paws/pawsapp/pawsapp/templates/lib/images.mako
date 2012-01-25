% for im in images:
<div class="span16">
  <p><span class="label notice">Function '${im["fu"]}'</span></p>
  <div>
    <a href="${im['link']}" class="ZOOM_IMAGE"><img src="${im['image']}"/></a>
  </div>
  <br clear="all"/>
  % if images.index(im) != len(images)-1:
  <hr/>
  % endif 
</div>
% endfor
