% for img in images:
<div style="clear:both; width: 100%">
  <p><span class="label notice">Function '${img["fu"]}'</span></p>
  % if img["zoom"]:
  <a href="${img['full']}" class="ZOOM_IMAGE" title="Zoom">${h.image(img["thumb"], img["fu"])}</a>
  % else:
  ${h.image(img["full"], img["fu"])}
  % endif
</div>
% if images.index(img) != len(images)-1:
<hr/>
% endif 
% endfor
