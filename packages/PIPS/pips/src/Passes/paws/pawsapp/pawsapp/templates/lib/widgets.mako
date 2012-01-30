<%doc>
  Widgets for PAWS
</%doc>


## Modal panel
<%def name="modal(header, content, id)">
<div class="modal hide fade" id="${id}" style="display: none;">
  <div class="modal-header">
    <a class="close" href="#">×</a>
    <h3>
      % if callable(header):
      ${header() | n}
      % else:
      ${header | n}
      % endif
    </h3>
  </div>
  <div class="modal-body" >
    % if callable(content):
    ${content() | n}
    % else:
    ${content | n}
    % endif
  </div>
</div>
</%def>


## Source tab
<%def name="source_tab(title=u'SOURCE', id=1, active=False)">
<li class="${h.css_classes([('active', active)])}" id="source-${id}_tab">${h.link_to(title, url="#source-%s" % id)}</li>
</%def>


## Source panel
<%def name="source_panel(id=1, active=False)">
<div id="source-${id}" class="${h.css_classes([('tab_pane', True), ('active', active)])}">
  <form>
    <fieldset style="padding-top:0">
      <label for="lang-${id}">Language </label>
      <div class="input">
	<input id="lang-${id}" value="not yet detected." readonly="readonly"/>
      </div>
    </fieldset>
    <textarea id="sourcecode-${id}" class="span16" rows="27"
	      onkeydown="handle_keydown(this, event)">Put your source code here.</textarea>
  </form>
</div>
</%def>
