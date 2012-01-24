## Export selected routes to Javascript

var routes = {}
%for r in routes:
routes['${r["name"]}'] = '${r["pattern"]}';
%endfor
