<%doc>
  PAWS home page
</%doc>


<%inherit file="base.mako"/>


<%def name="pagetitle()">
PYPS DEMO PAGE
</%def>

<%def name="js_slot()">
${h.javascript_link(request.static_url("pawsapp:static/jq/bootstrap-twipsy-min.js"))}
${h.javascript_link(request.static_url("pawsapp:static/jq/bootstrap-popover-min.js"))}
<script type="text/javascript">
  $(function () {
% for s in sections:
% for t in s["tools"]:
    $("${'#%s-%s' % (s['path'], t['name'])}").popover({html:true, placement: "left"});
% endfor
% endfor
  });
</script>
</%def>


## LEFT COLUMN

<%def name="left_column()">
${text|n}
</%def>


## MAIN COLUMN

<%def name="main_column()">

<div class="hero-unit">
  <h1>${h.image(request.static_url("pawsapp:static/img/paws-small-trans.gif"), u"PAWS Logo")} PIPS as a Web Service</h1>
  <p>Vestibulum id ligula porta felis euismod semper. Integer posuere
    erat a ante venenatis dapibus posuere velit aliquet. Duis mollis,
    est non commodo luctus, nisi erat porttitor ligula, eget lacinia
    odio sem nec elit.</p>
  <p><a class="btn primary large" href="http://pips4u.org" target="_blank">Learn more »</a></p>
</div>

<div class="row">
  <p class="span11 offset5">
    <span class="label notice">Notice</span>
    Run the mouse over a heading to see the corresponding description.
  </p>
</div>

## Section
% for s in sections:

<hr/>

## Subsection
% for t in s["tools"]:
<% first = bool(s["tools"].index(t) == 0) %>
<div class="row">
  % if first:
  <div class="span5">
    <h2>${s["title"]}</h2>
  </div>
  % endif
  <div class="${h.css_classes([('span4', True), ('offset5', not first)])}">
    <h3 id="${s['path'] + '-' + t['name']}" title="${t['name'].upper()}" data-content="${t['descr']}">${t["name"].upper()}</h3>
  </div>
  <div class="span4">
    <b>${h.link_to(u"basic »", url="/%s/%s" % (s["path"], t["name"]), class_="btn primary")}</b>
    % if s["advmode"]:
    <b>${h.link_to(u"advanced »", url="/%s/%s/advanced" % (s["path"], t["name"]), class_="btn success")}</b>
    % endif
  </div>
</div>
% endfor

% if not s["tools"]:
<div class="span5">
  <h2>${s["title"]}</h2>
</div>
% endif

% endfor
</%def>
