<%doc>
  Widgets for PAWS
</%doc>


## Modal panel

<%def name="modal(header, content, id, icon=None)">
<div class="modal hide span6" id="${id}" style="display: none;">
  <div class="modal-header">
    <a class="close" data-dismiss="modal">×</a>
    <h3>
      % if icon:
      <i class="icon-${icon}"></i>
      % endif
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
  <div class="modal-footer" >
    <button class="pull-right btn" id="${id}-close-button">Close</button>
  </div>
</div>
<script type="text/javascript">
  $(function() {
    $("#${id}-close-button").click(function(ev) {
      ev.preventDefault();
      $("#${id}").modal("hide");
    });
  });
</script>
</%def>


## Icon
<%def name="icon(name, white=False)">
<i class="${h.css_classes([('icon-' + name, True), ('icon-white', white)])}"></i>
</%def>


## Source tab

<%def name="source_tab(title=u'SOURCE', id=1, active=False)">
<li class="${h.css_classes([('active', active)])}" id="source-${id}_tab"><a href="#source-${id}" data-toggle="tab">${title}</a></li>
</%def>


## Source panel

<%def name="source_panel(id=1, active=False)">
<div id="source-${id}" class="${h.css_classes([('tab-pane', True), ('active', active)])}">
  <p>Language : <span id="lang-${id}" class="label">not yet detected</span></p>
  <textarea id="sourcecode-${id}" class="span12" rows="27"
	    onkeydown="handle_keydown(this, event)">Put your source code here.</textarea>
</div>
</%def>
