/*
 * "Learning to think (again)" tab.
 *
 * Pulls the daily log and today's assignment live from the independent challenge
 * repo (raw.githubusercontent.com), so the site stays decoupled from it and the
 * highlighted challenge is always current with no site rebuild. Ships with a tiny
 * self-contained Markdown renderer (no external CDN, CSP-friendly).
 */
(function () {
  'use strict';

  var OWNER = 'julienbrandoit';
  var REPO = 'Learning-to-think-again';
  var BRANCH = 'main';
  var RAW = 'https://raw.githubusercontent.com/' + OWNER + '/' + REPO + '/' + BRANCH + '/';
  var VIEW = 'https://github.com/' + OWNER + '/' + REPO + '/blob/' + BRANCH + '/';
  var REPO_URL = 'https://github.com/' + OWNER + '/' + REPO;

  var todayEl = document.getElementById('challenge-today');
  var logEl = document.getElementById('challenge-log');
  if (!todayEl || !logEl) return;

  injectStyles();

  /* ------------------------------------------------------------------ *
   * Minimal Markdown renderer (headings, paragraphs, lists, blockquotes,
   * fenced code, tables, inline code/bold/italic/links). Enough for the
   * challenge README format; everything is HTML-escaped first.
   * ------------------------------------------------------------------ */

  function escapeHtml(s) {
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  function renderInline(text) {
    // Split off inline-code spans so their contents are never reformatted.
    var parts = text.split('`');
    var out = '';
    for (var i = 0; i < parts.length; i++) {
      if (i % 2 === 1) {
        out += '<code>' + escapeHtml(parts[i]) + '</code>';
      } else {
        out += formatText(escapeHtml(parts[i]));
      }
    }
    return out;
  }

  function formatText(s) {
    s = s.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    s = s.replace(/__([^_]+)__/g, '<strong>$1</strong>');
    s = s.replace(/\*([^*]+)\*/g, '<em>$1</em>'); // only *…*, to avoid mangling snake_case
    s = s.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, function (_m, label, url) {
      var abs = /^https?:|^mailto:|^#/.test(url) ? url : VIEW + url.replace(/^\.?\//, '');
      return '<a href="' + abs + '" target="_blank" rel="noopener">' + label + '</a>';
    });
    return s;
  }

  function isTableSep(line) {
    return /^\s*\|?[\s:|-]+\|?\s*$/.test(line) && line.indexOf('-') !== -1;
  }

  function splitRow(line) {
    var cells = line.trim().replace(/^\|/, '').replace(/\|$/, '').split('|');
    return cells.map(function (c) { return c.trim(); });
  }

  function renderMarkdown(md) {
    var lines = md.replace(/\r\n/g, '\n').split('\n');
    var html = '';
    var i = 0;
    while (i < lines.length) {
      var line = lines[i];

      // fenced code block
      var fence = line.match(/^\s*(`{3,}|~{3,})(.*)$/);
      if (fence) {
        var marker = fence[1][0];
        var lang = fence[2].trim().replace(/[^A-Za-z0-9_-]/g, '');
        var buf = [];
        i++;
        while (i < lines.length && !(new RegExp('^\\s*' + marker + '{3,}\\s*$')).test(lines[i])) {
          buf.push(lines[i]);
          i++;
        }
        i++; // skip closing fence
        var cls = lang ? ' class="language-' + lang + '"' : '';
        html += '<pre><code' + cls + '>' + escapeHtml(buf.join('\n')) + '</code></pre>';
        continue;
      }

      // blank
      if (/^\s*$/.test(line)) { i++; continue; }

      // heading
      var h = line.match(/^(#{1,6})\s+(.*)$/);
      if (h) {
        var level = h[1].length;
        html += '<h' + level + '>' + renderInline(h[2].trim()) + '</h' + level + '>';
        i++;
        continue;
      }

      // horizontal rule
      if (/^\s*([-*_])(\s*\1){2,}\s*$/.test(line)) { html += '<hr>'; i++; continue; }

      // table
      if (/^\s*\|.*\|\s*$/.test(line) && i + 1 < lines.length && isTableSep(lines[i + 1])) {
        var header = splitRow(line);
        i += 2;
        var body = [];
        while (i < lines.length && /^\s*\|.*\|\s*$/.test(lines[i])) {
          body.push(splitRow(lines[i]));
          i++;
        }
        html += '<table><thead><tr>' +
          header.map(function (c) { return '<th>' + renderInline(c) + '</th>'; }).join('') +
          '</tr></thead><tbody>' +
          body.map(function (r) {
            return '<tr>' + r.map(function (c) { return '<td>' + renderInline(c) + '</td>'; }).join('') + '</tr>';
          }).join('') +
          '</tbody></table>';
        continue;
      }

      // blockquote
      if (/^\s*>/.test(line)) {
        var quote = [];
        while (i < lines.length && /^\s*>/.test(lines[i])) {
          quote.push(lines[i].replace(/^\s*>\s?/, ''));
          i++;
        }
        html += '<blockquote>' + renderMarkdown(quote.join('\n')) + '</blockquote>';
        continue;
      }

      // lists (unordered / ordered)
      var ulMatch = line.match(/^\s*[-*+]\s+/);
      var olMatch = line.match(/^\s*\d+\.\s+/);
      if (ulMatch || olMatch) {
        var ordered = !!olMatch;
        var tag = ordered ? 'ol' : 'ul';
        var items = [];
        var itemRe = ordered ? /^\s*\d+\.\s+(.*)$/ : /^\s*[-*+]\s+(.*)$/;
        while (i < lines.length && itemRe.test(lines[i])) {
          items.push(lines[i].match(itemRe)[1]);
          i++;
        }
        html += '<' + tag + '>' +
          items.map(function (it) { return '<li>' + renderInline(it) + '</li>'; }).join('') +
          '</' + tag + '>';
        continue;
      }

      // paragraph: gather consecutive plain lines
      var para = [];
      while (
        i < lines.length &&
        !/^\s*$/.test(lines[i]) &&
        !/^\s*(`{3,}|~{3,})/.test(lines[i]) &&
        !/^(#{1,6})\s+/.test(lines[i]) &&
        !/^\s*>/.test(lines[i]) &&
        !/^\s*[-*+]\s+/.test(lines[i]) &&
        !/^\s*\d+\.\s+/.test(lines[i]) &&
        !/^\s*\|.*\|\s*$/.test(lines[i])
      ) {
        para.push(lines[i].trim());
        i++;
      }
      html += '<p>' + renderInline(para.join(' ')) + '</p>';
    }
    return html;
  }

  /* ------------------------------------------------------------------ *
   * Daily-log parsing
   * ------------------------------------------------------------------ */

  function parseLog(readme) {
    var rows = [];
    var lines = readme.split('\n');
    for (var i = 0; i < lines.length; i++) {
      var l = lines[i];
      if (l.indexOf('|') === -1) continue;
      var cells = splitRow(l);
      if (cells.length < 6) continue;
      if (!/^\d{4}-\d{2}-\d{2}$/.test(cells[0])) continue; // skip header/separator
      var link = cells[3].match(/\[([^\]]+)\]\(([^)]+)\)/);
      if (!link) continue;
      var path = link[2].replace(/^\.?\//, '');
      rows.push({
        date: cells[0],
        id: cells[1],
        topic: cells[2],
        title: link[1],
        path: path,
        rawUrl: RAW + path,
        viewUrl: VIEW + path,
        difficulty: cells[4],
        done: /[☑✔xX]/.test(cells[5])
      });
    }
    return rows;
  }

  function todayISO() {
    var d = new Date();
    var m = String(d.getMonth() + 1).padStart(2, '0');
    var day = String(d.getDate()).padStart(2, '0');
    return d.getFullYear() + '-' + m + '-' + day;
  }

  /* ------------------------------------------------------------------ *
   * Rendering
   * ------------------------------------------------------------------ */

  function renderLog(rows, today) {
    if (!rows.length) {
      logEl.setAttribute('data-state', 'empty');
      logEl.innerHTML = '<p class="challenge-muted">The log is empty for now.</p>';
      return;
    }
    var body = rows.map(function (r) {
      var cls = r.date === today ? ' class="challenge-row-today"' : '';
      return '<tr' + cls + '>' +
        '<td>' + r.date + '</td>' +
        '<td>' + r.id + '</td>' +
        '<td>' + r.topic + '</td>' +
        '<td><a href="' + r.viewUrl + '" target="_blank" rel="noopener">' + escapeHtml(r.title) + '</a></td>' +
        '<td>' + r.difficulty + '</td>' +
        '<td>' + (r.done ? '☑' : '☐') + '</td>' +
        '</tr>';
    }).join('');
    logEl.setAttribute('data-state', 'ready');
    logEl.innerHTML =
      '<table><thead><tr><th>Date</th><th>ID</th><th>Topic</th><th>Challenge</th>' +
      '<th>Difficulty</th><th>Done</th></tr></thead><tbody>' + body + '</tbody></table>';
  }

  function renderTodayEmpty(rows, today) {
    var upcoming = rows.filter(function (r) { return r.date > today; })
      .sort(function (a, b) { return a.date < b.date ? -1 : 1; })[0];
    var past = rows.filter(function (r) { return r.date <= today; })
      .sort(function (a, b) { return a.date > b.date ? -1 : 1; })[0];
    var msg = '<p>No challenge scheduled for today (' + today + ').</p>';
    if (upcoming) {
      msg += '<p class="challenge-muted">Next up on ' + upcoming.date + ': ' +
        '<a href="' + upcoming.viewUrl + '" target="_blank" rel="noopener">' +
        escapeHtml(upcoming.title) + '</a>.</p>';
    } else if (past) {
      msg += '<p class="challenge-muted">Most recent: ' +
        '<a href="' + past.viewUrl + '" target="_blank" rel="noopener">' +
        escapeHtml(past.title) + '</a> (' + past.date + ').</p>';
    }
    todayEl.setAttribute('data-state', 'empty');
    todayEl.innerHTML = msg;
  }

  function renderTodayChallenge(row, markdown) {
    var body = markdown;
    var title = row.title;
    var firstHeading = body.match(/^\s*#\s+(.+)\s*$/m);
    if (firstHeading) {
      title = firstHeading[1].trim();
      body = body.replace(firstHeading[0], ''); // avoid a duplicate title
    }
    todayEl.setAttribute('data-state', 'ready');
    todayEl.innerHTML =
      '<div class="challenge-card-head">' +
      '<h3 style="margin:0">' + escapeHtml(title) + '</h3>' +
      '<a class="challenge-view-btn" href="' + row.viewUrl + '" target="_blank" rel="noopener">View on GitHub ↗</a>' +
      '</div>' +
      '<p class="challenge-badge">Today · ' + row.date + ' · ' + escapeHtml(row.topic) +
      ' · ' + row.difficulty + '</p>' +
      renderMarkdown(body);
  }

  function fail(message) {
    todayEl.setAttribute('data-state', 'error');
    todayEl.innerHTML = '<p>' + message + '</p>' +
      '<p class="challenge-muted">You can always read everything directly in the ' +
      '<a href="' + REPO_URL + '" target="_blank" rel="noopener">challenge repository</a>.</p>';
    if (logEl.getAttribute('data-state') === 'loading') {
      logEl.setAttribute('data-state', 'error');
      logEl.innerHTML = '<p class="challenge-muted">Could not load the log. ' +
        'See the <a href="' + REPO_URL + '" target="_blank" rel="noopener">repository</a>.</p>';
    }
  }

  /* ------------------------------------------------------------------ *
   * Boot
   * ------------------------------------------------------------------ */

  fetch(RAW + 'README.md', { cache: 'no-cache' })
    .then(function (r) {
      if (!r.ok) throw new Error('HTTP ' + r.status);
      return r.text();
    })
    .then(function (readme) {
      var today = todayISO();
      var rows = parseLog(readme);
      renderLog(rows, today);

      var match = null;
      for (var i = 0; i < rows.length; i++) {
        if (rows[i].date === today) { match = rows[i]; break; }
      }
      if (!match) { renderTodayEmpty(rows, today); return; }

      return fetch(match.rawUrl, { cache: 'no-cache' })
        .then(function (r) {
          if (!r.ok) throw new Error('HTTP ' + r.status);
          return r.text();
        })
        .then(function (md) { renderTodayChallenge(match, md); });
    })
    .catch(function (err) {
      fail("Could not load today's challenge (" + (err && err.message ? err.message : 'network error') + ').');
    });

  /* ------------------------------------------------------------------ */

  function injectStyles() {
    var css =
      '.challenge-today{border:1px solid var(--card-border-color,rgba(128,128,128,.25));' +
      'border-left:4px solid #d0603a;border-radius:8px;padding:1rem 1.25rem;margin:1rem 0 1.5rem;' +
      'background:rgba(208,96,58,.06)}' +
      '.challenge-today[data-state="empty"]{border-left-color:#9aa0a6;background:rgba(128,128,128,.06)}' +
      '.challenge-today[data-state="error"]{border-left-color:#c0392b;background:rgba(192,57,43,.06)}' +
      '.challenge-today>*:first-child{margin-top:0}.challenge-today>*:last-child{margin-bottom:0}' +
      '.challenge-card-head{display:flex;justify-content:space-between;align-items:baseline;' +
      'gap:.75rem 1rem;flex-wrap:wrap;margin-bottom:.35rem}' +
      '.challenge-view-btn{white-space:nowrap;font-size:.85rem;font-weight:600}' +
      '.challenge-badge{font-size:.75rem;text-transform:uppercase;letter-spacing:.05em;opacity:.7;margin:.15rem 0 .75rem}' +
      '.challenge-muted{opacity:.7}' +
      '.challenge-log{overflow-x:auto}.challenge-log table{width:100%}' +
      '.challenge-row-today{font-weight:600}' +
      '.challenge-row-today td:first-child{box-shadow:inset 3px 0 0 #d0603a}';
    var tag = document.createElement('style');
    tag.textContent = css;
    document.head.appendChild(tag);
  }
})();
