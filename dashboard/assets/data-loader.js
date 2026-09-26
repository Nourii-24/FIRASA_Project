// Firasa dashboard - loads the dashboard data as plain row objects.
// Over HTTP it reads data/*.csv. Opened straight from disk (file://) the browser
// blocks that, so it reads data/offline-data.js instead (same data, built by
// build_offline_data.py). Calls back with [] when nothing could be loaded.
var FirasaData = (function() {
  var offline = null, waiting = null;

  function loadOffline(cb) {
    if (offline) { cb(offline); return; }
    if (waiting) { waiting.push(cb); return; }
    waiting = [cb];
    function done() {
      offline = window.FIRASA_OFFLINE || {};
      var w = waiting; waiting = null;
      w.forEach(function(f) { f(offline); });
    }
    var s = document.createElement('script');
    s.src = 'data/offline-data.js';
    s.onload = done; s.onerror = done;
    document.head.appendChild(s);
  }

  function offlineRows(name, d) {
    if (name === 'customer_scores_with_shap' && d.scores) {
      var cols = d.scores.cols;
      return d.scores.rows.map(function(r) {
        var o = {};
        cols.forEach(function(c, i) { o[c] = r[i]; });
        return o;
      });
    }
    if (name === 'customer_trajectories' && d.trajectories) {
      var rows = [];
      Object.keys(d.trajectories).forEach(function(id) {
        d.trajectories[id].forEach(function(p, i) {
          rows.push({ customer_ID: id, month_index: i + 1, risk_score: p });
        });
      });
      return rows;
    }
    return [];
  }

  function load(name, cb) {
    if (location.protocol === 'file:') {
      loadOffline(function(d) { cb(offlineRows(name, d)); });
      return;
    }
    if (typeof Papa === 'undefined') { cb([]); return; }
    try {
      Papa.parse('data/' + name + '.csv', {
        download: true, header: true, skipEmptyLines: true,
        complete: function(res) { cb(res.data || []); },
        error: function() { cb([]); }
      });
    } catch (e) { cb([]); }
  }

  return { load: load };
})();
