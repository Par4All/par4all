<%doc>
  Generic page for a tool
</%doc>


<%inherit file="base.mako"/>

<%namespace name="w" file="pawsapp:templates/lib/widgets.mako"/>

<%def name="css_slot()">
${h.stylesheet_link(request.static_url("pawsapp:static/css/jq/jquery.jqzoom-min.css"), media="all")}
</%def>

<%def name="js_slot()">
${h.javascript_link(request.static_url("pawsapp:static/jq/jquery.jqzoom-core-pack-min.js"))}
<script type="text/javascript">
  operation = "${tutorial}";
  $(function () {
  $(".pagination li a").popover({html:true, placement: "bottom"});
  });
</script>
${h.javascript_link(request.static_url("pawsapp:static/js/tutorial.js"))}
</%def>




## ONE COLUMN

<%def name="content()">

<div class="container-fluid">

  <div class="hero-unit" style="padding:.5em 1em">
    <h2>${h.image(request.static_url("pawsapp:static/img/favicon-trans.gif"), u"PAWS icon")}
      ${info["title"]}
    </h2>
    ${info["descr"]|n}
  </div>

  <div id="pagination-container">
    % if step==0 and not initialized:
    <div class="pagination">
      <button class="btn btn-primary" id="run-button">${w.icon("play", True)} Start tutorial</button>
    </div>
    % else:
    ${w.tutorial_paginator()}
    % endif
  </div>

  <div id="demo" class="row-fluid">
    <div class="span6">
      % if images:
      ${w.images_page(images)}
      % else:
      <span class="label label-info">SOURCE</span>
      % if step == 0:
      <textarea name="sourcecode" id="sourcecode" class="span12" rows="34">${source}</textarea>
      % else:
      ${source|n}
      % endif
      <button class="btn">${w.icon("download-alt")} Save</button>
      <button class="btn">${w.icon("print")} Print</button>
      % endif
    </div>
    <div class="span6">
      <div>
	<span class="label label-info">SCRIPT</span>
      </div>
      <div id="sourcetpips" style="overflow:auto">
	${tpips|n}
      </div>
      <button class="btn">${w.icon("download-alt")} Save</button>
      <button class="btn">${w.icon("print")} Print</button>
    </div>
  </div>

</div>
</%def>

