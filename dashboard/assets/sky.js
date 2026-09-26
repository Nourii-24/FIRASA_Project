// Firasa dashboard - twinkling sky background.
// Fills <div class="sky" id="sky"> with stars, sparkles and shooting stars.
// Optional data-density="0..1" on that div thins the effect out.
(function () {
  var sky = document.getElementById('sky');
  if (!sky) return;
  var colors = ['#E9CB6F', '#E9CB6F', '#F3EFE3', '#3FC98A', '#D4AF37'];
  function rnd(a, b) { return a + Math.random() * (b - a); }
  function pick(list) { return list[Math.floor(Math.random() * list.length)]; }
  var density = parseFloat(sky.getAttribute('data-density')) || 1;
  var area = window.innerWidth * window.innerHeight * density;
  var html = '';
  var stars = Math.min(140, Math.round(area / 11000));
  for (var i = 0; i < stars; i++) {
    html += '<span class="sky-st" style="left:' + rnd(0, 100).toFixed(2) + '%;top:' + rnd(0, 100).toFixed(2) + '%;--s:' + rnd(1, 2.6).toFixed(1) + 'px;--c:' + pick(colors) +
            ';--o:' + rnd(0.5, 1).toFixed(2) + ';--t:' + rnd(2.5, 6).toFixed(1) + 's;--d:-' + rnd(0, 6).toFixed(1) + 's"></span>';
  }
  var sparkles = Math.min(22, Math.round(area / 70000));
  for (var j = 0; j < sparkles; j++) {
    html += '<span class="sky-sp" style="left:' + rnd(2, 96).toFixed(2) + '%;top:' + rnd(4, 94).toFixed(2) + '%;--s:' + Math.round(rnd(12, 30)) + 'px;--c:' + pick(['#E9CB6F', '#D4AF37', '#3FC98A']) +
            ';--t:' + rnd(3.5, 7).toFixed(1) + 's;--d:' + rnd(0, 7).toFixed(1) + 's"><i></i></span>';
  }
  for (var k = 0; k < (density < 1 ? 1 : 3); k++) {
    html += '<span class="sky-shoot" style="left:' + rnd(5, 60).toFixed(1) + '%;top:' + rnd(5, 45).toFixed(1) + '%;--t:' + rnd(9, 16).toFixed(1) + 's;--d:' + rnd(1, 12).toFixed(1) + 's"></span>';
  }
  sky.innerHTML = html;
})();
