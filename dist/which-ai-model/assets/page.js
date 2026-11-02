/* Pure selection over recorded measurements. The saved benchmark rule is unchanged. */
function selectScorePick(rows, minimum, priority) {
  function known(row, key) {
    return row[key] && typeof row[key].value === 'number' && Number.isFinite(row[key].value) && row[key].value >= 0;
  }
  function cost(row) { return known(row, 'cost') ? row.cost.value : Infinity; }
  function tie(a, b) { return cost(a) - cost(b) || a.model.localeCompare(b.model); }
  if (!Number.isFinite(minimum) || minimum < 0 || minimum > 1) return { candidate: null, qualifying: 0, reason: 'score' };
  var eligible = rows.filter(function (row) { return known(row, 'metric') && row.metric.value <= 1 && row.metric.value + 1e-12 >= minimum; });
  var candidates = eligible.filter(function (row) {
    return priority === 'cheapest' ? known(row, 'cost') : priority === 'fastest' ? known(row, 'latency') : true;
  });
  candidates.sort(function (a, b) {
    if (priority === 'cheapest') return tie(a, b);
    if (priority === 'fastest') return a.latency.value - b.latency.value || tie(a, b);
    return b.metric.value - a.metric.value || tie(a, b);
  });
  return { candidate: candidates[0] || null, qualifying: eligible.length, reason: eligible.length ? (priority === 'fastest' ? 'latency' : 'cost') : 'score' };
}

/* Recorded evidence only. No requests, dependencies or model calls. */
(function () {
  'use strict';
  if (typeof document === 'undefined') return;
  var root = document.documentElement;
  root.classList.add('js');
  var volumeInput = document.getElementById('volume-items');
  var volumeTable = document.getElementById('volume-table');
  var volumeHint = document.getElementById('volume-hint');
  var picks = document.getElementById('picks');
  var radios = Array.from(document.querySelectorAll('[data-pick]'));
  var live = document.getElementById('picker-live');
  var currentPriority = 'cheapest';
  var limits = Array.from(document.querySelectorAll('[data-min-score]'));
  var currentMinimum = 75;
  var explorerTasks = [];
  try { explorerTasks = JSON.parse(document.getElementById('picker-data').textContent); } catch (e) { /* Keep the recorded fallback. */ }
  var explorerReady = Array.isArray(explorerTasks) && explorerTasks.length > 0;
  var numberFormats = {};
  var currencyFormats = {};

  function words(en, de) { return root.lang === 'de' ? de : en; }
  function store(key, value) { try { localStorage.setItem(key, value); } catch (e) { /* Preferences are optional. */ } }
  function query(key, value) {
    try {
      var url = new URL(location.href);
      url.searchParams.set(key, value);
      history.replaceState(null, '', url);
    } catch (e) { /* file:// and restricted history still work without deep links. */ }
  }
  function selectedLabel() {
    var selected = radios.find(function (button) { return button.dataset.pick === currentPriority; });
    return selected ? selected.querySelector('[data-copy-lang="' + root.lang + '"]').textContent : '';
  }
  function announce() {
    live.textContent = words('Showing recommendations for: ', 'Empfehlungen für: ') + selectedLabel() +
      (explorerReady ? words('. Minimum score: ', '. Mindestwert: ') + currentMinimum + '%' : '') + '.';
  }
  function renderExplorer() {
    if (!explorerReady) return;
    explorerTasks.forEach(function (task) {
      var row = picks.querySelector('[data-task="' + task.id + '"]');
      if (!row) return;
      var result = selectScorePick(task.rows, currentMinimum / 100, currentPriority);
      var panel = row.querySelector('.explorer-pick');
      var empty = row.querySelector('[data-empty-pick]');
      var note = row.querySelector('[data-qualification-note]');
      panel.hidden = !result.candidate;
      empty.hidden = Boolean(result.candidate);
      note.hidden = !result.candidate || result.qualifying !== 1;
      note.textContent = words('The same choice for every priority. The other models score below your limit.', 'Bei jeder Priorität dieselbe Wahl. Die anderen Modelle liegen unter Ihrem Mindestwert.');
      if (result.candidate) {
        var selected = result.candidate;
        panel.querySelector('[data-selected-model]').textContent = selected.name;
        panel.querySelector('[data-selected-score]').textContent = formatEvidence(selected.metric.display);
        panel.querySelector('[data-selected-cost]').textContent = selected.cost && selected.cost.value !== null ? formatEvidence(selected.cost.display) : words('Unknown', 'Unbekannt');
        panel.querySelector('[data-selected-latency]').textContent = selected.latency && selected.latency.value !== null ? formatEvidence(selected.latency.display) : words('Unknown', 'Unbekannt');
        panel.querySelector('[data-qualifying-badge]').hidden = result.qualifying !== 1;
      } else if (result.reason === 'score') {
        empty.textContent = words('No model reaches ', 'Kein Modell erreicht ') + currentMinimum + words('% on this task. Try a lower limit.', '% bei dieser Aufgabe. Wählen Sie einen niedrigeren Mindestwert.');
      } else {
        empty.textContent = result.reason === 'latency'
          ? words('No recorded response time is available for the qualifying models. Choose another priority.', 'Für die passenden Modelle ist keine Laufzeit aufgezeichnet. Wählen Sie eine andere Priorität.')
          : words('No recorded cost is available for the qualifying models. Choose another priority.', 'Für die passenden Modelle sind keine Kosten aufgezeichnet. Wählen Sie eine andere Priorität.');
      }
    });
  }
  function applyTheme(theme, persist) {
    root.dataset.theme = theme;
    document.querySelector('meta[name="theme-color"]').content = theme === 'dark' ? '#030712' : '#f9fafb';
    document.querySelectorAll('[data-theme-set]').forEach(function (button) {
      button.setAttribute('aria-pressed', String(button.dataset.themeSet === theme));
    });
    if (persist) { store('wam-theme', theme); query('theme', theme); }
  }
  document.querySelectorAll('[data-theme-set]').forEach(function (button) {
    button.addEventListener('click', function () { applyTheme(button.dataset.themeSet, true); });
  });

  // Preserve evidence precision; only separators change with the chosen language.
  var numericNodes = Array.from(document.querySelectorAll('.pick-stats dd, .facts-strip strong, .baseline-stats dd, .large-number, table td:not([data-live]), .fixed-volume, .floor-note strong, .gate-result p strong, .gate-copy p strong, .example-row:not(.head) > span:nth-child(3), .example-row:not(.head) > span:nth-child(4)')).filter(function (node) { return !node.closest('.explorer-pick'); });
  numericNodes.forEach(function (node) { node.dataset.original = node.textContent; });
  function formatEvidence(text) {
    if (root.lang !== 'de') return text;
    return text.replace(/\d[\d,]*(?:\.\d+)?/g, function (token) {
      var decimals = (token.split('.')[1] || '').length;
      if (!numberFormats[decimals]) numberFormats[decimals] = new Intl.NumberFormat('de-DE', { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
      return numberFormats[decimals].format(Number(token.replace(/,/g, '')));
    }).replace(' CI ', ' KI ').replace(/ to /g, ' bis ');
  }
  function updateVolume() {
    if (!volumeInput || !volumeTable) return;
    var items = volumeInput.valueAsNumber;
    var valid = Number.isSafeInteger(items) && items >= 1 && items <= 1000000000;
    volumeInput.setAttribute('aria-invalid', String(!valid));
    volumeTable.querySelectorAll('[data-cost]').forEach(function (row) {
      var cell = row.querySelector('[data-live]');
      var unit = Number(row.dataset.cost);
      if (!valid || !Number.isFinite(unit)) { cell.textContent = '—'; return; }
      var cost = unit * items / 1000;
      var digits = cost === 0 ? 0 : cost < .0001 ? 6 : cost < .001 ? 5 : cost < .01 ? 4 : 2;
      var key = root.lang + digits;
      if (!currencyFormats[key]) currencyFormats[key] = new Intl.NumberFormat(root.lang === 'de' ? 'de-DE' : 'en-US', {
        style: 'currency', currency: 'USD', minimumFractionDigits: digits, maximumFractionDigits: digits
      });
      cell.textContent = currencyFormats[key].format(cost);
    });
    volumeHint.textContent = valid
      ? words('Estimated monthly cost at ', 'Geschätzte Monatskosten bei ') + new Intl.NumberFormat(root.lang).format(items) + words(' items.', ' Aufgaben.')
      : words('Enter a whole number from 1 to 1,000,000,000.', 'Eine ganze Zahl von 1 bis 1.000.000.000 eingeben.');
  }
  function applyLanguage(lang, persist) {
    root.lang = lang;
    document.querySelectorAll('[data-lang]').forEach(function (button) {
      button.setAttribute('aria-pressed', String(button.dataset.lang === lang));
    });
    document.querySelectorAll('[data-label-en]').forEach(function (node) {
      node.setAttribute('aria-label', lang === 'de' ? node.dataset.labelDe : node.dataset.labelEn);
    });
    numericNodes.forEach(function (node) {
      var formatted = formatEvidence(node.dataset.original);
      if (node.textContent !== formatted) node.textContent = formatted;
    });
    document.title = words('Which AI Model? — Saber Sojudi', 'Which AI Model? — ein Projekt von Saber Sojudi');
    var description = words('Which model fits each job? A recorded AI benchmark with a task picker, cost comparison and clear limits. A project by Saber Sojudi.', 'Welches KI-Modell passt zu welcher Aufgabe? Aufgezeichnete Ergebnisse, Modellauswahl, Kostenvergleich und klare Grenzen. Ein Projekt von Saber Sojudi.');
    document.querySelector('meta[name="description"]').content = description;
    document.querySelector('meta[property="og:description"]').content = description;
    if (persist) { store('wam-lang', lang); query('lang', lang); }
    updateVolume();
    renderExplorer();
    announce();
  }
  document.querySelectorAll('[data-lang]').forEach(function (button) {
    button.addEventListener('click', function () { applyLanguage(button.dataset.lang, true); });
  });
  function selectPriority(button, focus, persist) {
    currentPriority = button.dataset.pick;
    picks.dataset.mode = currentPriority;
    radios.forEach(function (other) {
      var active = button === other;
      other.setAttribute('aria-checked', String(active));
      other.tabIndex = active ? 0 : -1;
    });
    if (persist) query('priority', currentPriority);
    renderExplorer();
    announce();
    if (focus) button.focus();
  }
  radios.forEach(function (button, index) {
    button.addEventListener('click', function () { selectPriority(button, false, true); });
    button.addEventListener('keydown', function (event) {
      var next;
      if (event.key === 'ArrowRight' || event.key === 'ArrowDown') next = (index + 1) % radios.length;
      if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') next = (index + radios.length - 1) % radios.length;
      if (event.key === 'Home') next = 0;
      if (event.key === 'End') next = radios.length - 1;
      if (next !== undefined) { event.preventDefault(); selectPriority(radios[next], true, true); }
    });
  });
  function selectMinimum(button, focus, persist) {
    currentMinimum = Number(button.dataset.minScore);
    limits.forEach(function (other) {
      var active = other === button;
      other.setAttribute('aria-checked', String(active));
      other.tabIndex = active ? 0 : -1;
    });
    if (persist) query('min_score', String(currentMinimum));
    renderExplorer();
    announce();
    if (focus) button.focus();
  }
  limits.forEach(function (button, index) {
    button.addEventListener('click', function () { selectMinimum(button, false, true); });
    button.addEventListener('keydown', function (event) {
      var next;
      if (event.key === 'ArrowRight' || event.key === 'ArrowDown') next = (index + 1) % limits.length;
      if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') next = (index + limits.length - 1) % limits.length;
      if (event.key === 'Home') next = 0;
      if (event.key === 'End') next = limits.length - 1;
      if (next !== undefined) { event.preventDefault(); selectMinimum(limits[next], true, true); }
    });
  });
  picks.classList.add('has-js');
  if (explorerReady) { root.classList.add('score-explorer'); picks.classList.add('is-exploring'); }
  if (volumeInput && volumeTable) {
    volumeTable.classList.add('has-js');
    document.getElementById('volume-nojs').hidden = true;
    volumeInput.addEventListener('input', updateVolume);
  }
  var requested = new URLSearchParams(location.search).get('priority');
  var requestedMinimum = new URLSearchParams(location.search).get('min_score');
  selectMinimum(limits.find(function (button) { return button.dataset.minScore === requestedMinimum; }) || limits.find(function (button) { return button.dataset.minScore === '75'; }), false, false);
  selectPriority(radios.find(function (button) { return button.dataset.pick === requested; }) || radios[0], false, false);
  applyTheme(root.dataset.theme || 'light', false);
  applyLanguage(root.lang === 'de' ? 'de' : 'en', false);

  // Ordinary anchors retain native hash links and Back/Forward behavior.
  var links = Array.from(document.querySelectorAll('.nav a'));
  var sections = links.map(function (link) { return document.querySelector(link.hash); });
  function markNavigation() {
    var threshold = parseInt(getComputedStyle(root).getPropertyValue('--header-offset'), 10) + 40;
    var active = sections[0];
    sections.forEach(function (section) { if (section.getBoundingClientRect().top <= threshold) active = section; });
    links.forEach(function (link) {
      if (link.hash === '#' + active.id) link.setAttribute('aria-current', 'location');
      else link.removeAttribute('aria-current');
    });
  }
  var scheduled = false;
  addEventListener('scroll', function () {
    if (scheduled) return;
    scheduled = true;
    requestAnimationFrame(function () { markNavigation(); scheduled = false; });
  }, { passive: true });
  addEventListener('hashchange', markNavigation);
  markNavigation();
})();
