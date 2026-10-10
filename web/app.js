// Chess Lab UI. The page passes a backend with
//   backend.score(moves, prompt) -> [{uci, san, logp}] for every legal move (moves = the game so far, UCI)
//   backend.key() -> a string naming the current runner (results are cached per runner)
// play.html scores either in the browser (ONNX Runtime Web) or on your own PyTorch server.

const PROMPTS = [
  'castle queenside and storm the enemy king with your pawns',
  'go for the Sicilian Defense',
  'solid, safe, risk-free chess: wait for your opponent to make mistakes',
  'leave your king in the center and go for an open fight',
  'trade queens as early as possible',
  'bring your queen out early and push random pawns for no reason',
];

const colorBox = c => `
  <div id="box_${c}" class="color-box">
    <h4>${c[0].toUpperCase() + c.slice(1)}
      <select id="who_${c}" class="who"><option value="human">You</option><option value="engine">Engine</option></select></h4>
    <label>Style prompt (empty = no prompt)<textarea id="prompt_${c}" placeholder="e.g. castle queenside and storm the enemy king"></textarea></label>
    <div class="chips">${PROMPTS.map(p => `<button class="chip" data-color="${c}" data-prompt="${p}">${p}</button>`).join('')}</div>
    <label>Temperature (0 = always the most likely move) <input id="temp_${c}" type="range" min="0" max="1.5" step="0.05" value="0">
      <span id="tempVal_${c}">0</span></label>
  </div>`;

// Builds the page; `top` is page-specific HTML shown above the board (backend status).
export function mount(page, top) {
  document.getElementById('app').innerHTML = `
  <header>
    <div><h1>CHSM8</h1><p class="subtitle">A chess model that plays in the style you describe. Give each side a prompt,
      let the engine move for either side, and click any piece to see the move probabilities behind its choice.</p></div>
    <nav><a href="index.html">Project</a><a href="play.html" class="on">Play</a><a href="https://huggingface.co/LegumMagister/chsm8">Model</a></nav>
  </header>
  <div class="backend" id="backendInfo">${top}</div>
  <div class="row">
    <div>
      <div id="boardWrap"><div id="board"></div><div id="moveHints"></div><div id="thinkingOverlay"></div><div id="promoPopup"></div></div>
      <div id="status">loading…</div>
      <div>
        <button id="probsBtn">Show probabilities</button><button id="engineMoveBtn" class="primary">Let engine move</button><button id="autoplayBtn">Engine vs engine</button>
      </div>
      <div>
        <label class="inline"><input type="checkbox" id="autoEngine" checked> engine replies automatically after your move</label>
      </div>
      <button id="newGameBtn">New game</button><button id="undoBtn">Undo</button><button id="flipBtn">Flip board</button>
      <fieldset><legend>Move probabilities for the side to move</legend>
        <div class="note">"Show probabilities" colours each piece by the chance that it moves next. Click a piece to see where it goes.</div>
        <label><input type="checkbox" id="distGlobal"> clicked piece: share of all legal moves (default: its own moves sum to 1)</label>
        <div id="heatmapLegend"></div><div id="distPanel"></div></fieldset>
    </div>
    <div class="side">
      ${colorBox('white')}${colorBox('black')}
      <h4>Moves</h4><div id="log"></div>
    </div>
  </div>`;
}

export function startApp(backend) {
  const game = new Chess();
  let modelReady = false;
  let board = null, busy = false, promoting = false, autoplay = false, forceOnce = false;
  const cache = new Map();
  const moveLog = [];                                    // one line per ply, so Undo can drop lines
  const $id = id => document.getElementById(id);
  const turnColor = () => (game.turn() === 'w' ? 'white' : 'black');
  const otherColor = c => (c === 'white' ? 'black' : 'white');
  const cap = c => c[0].toUpperCase() + c.slice(1);
  const isEngine = c => $id('who_' + c).value === 'engine';
  const temperature = c => parseFloat($id('temp_' + c).value);
  const promptOf = c => $id('prompt_' + c).value.trim();
  const history = () => game.history({ verbose: true }).map(m => m.from + m.to + (m.promotion || '')).join(' ');
  const state = () => backend.key() + '|' + history() + '|' + promptOf(turnColor());   // what a shown result belongs to
  const sleep = ms => new Promise(r => setTimeout(r, ms));

  function setStatus(t) { $id('status').textContent = t; }
  function setBusy(b, msg) {
    busy = b;
    $id('thinkingOverlay').style.display = b ? 'flex' : 'none';
    $id('thinkingOverlay').textContent = msg || 'Thinking…';
    for (const id of ['newGameBtn', 'undoBtn', 'engineMoveBtn', 'probsBtn']) $id(id).disabled = b;
  }

  // Readable errors: a browser fetch to a dead server only says "Failed to fetch".
  function friendly(e) {
    const msg = (e && e.message) || String(e);
    if (/failed to fetch|networkerror|load failed|network request failed/i.test(msg))
      return 'the server is not reachable. Start it with "python server.py" and check the address, or use "In this browser".';
    return msg;
  }

  // Log-probs of every legal move for the side to move under that side's current prompt, cached.
  async function scores() {
    const moves = history(), prompt = promptOf(turnColor()), key = backend.key() + '|' + moves + '|' + prompt;
    if (!cache.has(key)) cache.set(key, await backend.score(moves, prompt));
    return cache.get(key);
  }

  function softmax(moves, t) {
    t = Math.max(t, 1e-3);
    const m = Math.max(...moves.map(x => x.logp));
    const e = moves.map(x => Math.exp((x.logp - m) / t));
    const z = e.reduce((a, b) => a + b, 0);
    return moves.map((x, i) => ({ ...x, p: e[i] / z }));
  }

  function pick(ps, t) {
    if (t <= 0) return ps.reduce((a, b) => (b.p > a.p ? b : a));
    let r = Math.random();
    for (const m of ps) { if ((r -= m.p) <= 0) return m; }
    return ps[ps.length - 1];
  }

  function gameOverText() {
    if (game.in_checkmate()) return `Checkmate: ${game.turn() === 'w' ? 'Black' : 'White'} wins.`;
    const kind = game.in_stalemate() ? 'stalemate' : game.in_threefold_repetition() ? 'threefold repetition'
               : game.insufficient_material() ? 'insufficient material' : 'the 50-move rule';
    return `Draw by ${kind}.`;
  }

  function renderLog() {
    const el = $id('log');
    el.textContent = moveLog.join('\n') + (game.game_over() ? `\n${gameOverText()}` : '');
    el.scrollTop = el.scrollHeight;
  }
  function log(line) { moveLog.push(line); renderLog(); }
  const short = p => (p ? `"${p.length > 40 ? p.slice(0, 40) + '…' : p}"` : 'no prompt');

  function markLastMove() {
    $('#board .last').removeClass('last');
    const h = game.history({ verbose: true }), m = h[h.length - 1];
    if (m) $('#board .square-' + m.from + ', #board .square-' + m.to).addClass('last');
  }
  function refresh() {
    board.position(game.fen());
    deselect();
    clearOverlays();
    markLastMove();
    updateBoxes();
    renderLog();
    if (game.game_over()) {
      stopAutoplay();
      setStatus(`Game over. ${gameOverText()} Press "New game" to play again, or "Undo" to take moves back.`); return;
    }
    const c = turnColor();
    if (autoplay) setStatus(`Engine vs engine: ${c} to move. Press "Stop" to pause.`);
    const chk = game.in_check() ? ' Check!' : '';
    if (!autoplay) setStatus(isEngine(c) ? `${cap(c)} to move (engine).${chk} Press "Let engine move", or "Show probabilities" first.`
                                         : `${cap(c)} to move (you).${chk} Drag or tap a piece, or let the engine move for you.`);
  }

  // Board faces the human: if exactly one side is played by you, that side is at the bottom.
  function syncOrientation() {
    const hw = !isEngine('white'), hb = !isEngine('black');
    const want = hw && !hb ? 'white' : hb && !hw ? 'black' : null;
    if (want && board.orientation() !== want) board.orientation(want);
  }

  // ---- engine turns ----
  // The engine moves on its own when: autoplay is running; or it was just given the move (New game with the
  // engine as White, a side switched to Engine); or it replies to your move with "replies automatically" on.
  function wantEngine() {
    if (busy || promoting || game.game_over() || !isEngine(turnColor())) return false;
    if (autoplay || forceOnce) return true;
    return !isEngine(otherColor(turnColor())) && $id('autoEngine').checked;
  }
  async function kick() {
    if (!wantEngine()) return;
    if (autoplay) { await sleep(300); if (!wantEngine()) return; }   // let the viewer see each move
    await engineMove();
  }

  async function engineMove() {
    if (busy || promoting || game.game_over()) return;
    const c = turnColor(), prompt = promptOf(c), before = history();
    forceOnce = false;
    setBusy(true, autoplay ? `${cap(c)} is thinking…` : 'Thinking…');
    let m, p;
    try {
      const moves = await scores();
      if (!moves || !moves.length) throw new Error('the model returned no moves for this position');
      const ps = softmax(moves, 1);                          // probabilities as the model gives them
      m = pick(softmax(moves, temperature(c)), temperature(c));
      p = ps.find(x => x.uci === m.uci).p;
    } catch (e) {
      setBusy(false); stopAutoplay(); refresh(); setStatus('Error: ' + friendly(e)); return;
    }
    setBusy(false);
    if (history() !== before) { refresh(); return; }         // position changed meanwhile: drop this result
    const mv = game.move({ from: m.uci.slice(0, 2), to: m.uci.slice(2, 4), promotion: m.uci[4] });
    if (!mv) { stopAutoplay(); refresh(); setStatus(`Error: the model chose an illegal move (${m.uci})`); return; }
    log(`${game.history().length}. ${c} (engine, ${short(prompt)}): ${mv.san}  p=${(p * 100).toFixed(1)}%`);
    refresh();
    kick();
  }

  function startAutoplay() {
    if (game.game_over()) { refresh(); return; }
    $id('who_white').value = 'engine'; $id('who_black').value = 'engine';
    autoplay = true; $id('autoplayBtn').textContent = 'Stop'; $id('autoplayBtn').classList.add('primary');
    refresh(); kick();
  }
  function stopAutoplay() {
    if (!autoplay) return;
    autoplay = false; $id('autoplayBtn').textContent = 'Engine vs engine'; $id('autoplayBtn').classList.remove('primary');
    if (!game.game_over()) setStatus(`Stopped. ${cap(turnColor())} to move.`);
  }

  // ---- your moves ----
  function onDragStart(source, piece) {
    if (busy || promoting || autoplay || game.game_over() || isEngine(turnColor())) return false;
    if (selected && targets(selected).includes(source)) { tryMove(selected, source); return false; }   // tap-to-capture
    if (piece[0] !== game.turn()) return false;             // only the side to move can be dragged
  }

  function humanMove(from, to, promotion) {
    const c = turnColor();
    deselect();
    const m = game.move({ from, to, promotion });
    if (m === null) { board.position(game.fen()); return; }
    forceOnce = false;
    log(`${game.history().length}. ${c} (you): ${m.san}`);
    setTimeout(() => { refresh(); kick(); }, 0);
  }

  const PROMO = { q: 'queen', r: 'rook', b: 'bishop', n: 'knight' };
  function askPromotion(color) {
    return new Promise(resolve => {
      const pop = $id('promoPopup'), w = color[0];
      pop.innerHTML = '<div class="promo-box"><div class="promo-title">Promote to</div><div>' +
        Object.entries(PROMO).map(([p, name]) => `<button data-p="${p}" title="${name}"><img src="vendor/img/chesspieces/wikipedia/${w}${p.toUpperCase()}.png" alt="${name}"></button>`).join('') +
        '</div><button data-p="" class="promo-cancel">Cancel</button></div>';
      pop.style.display = 'flex'; promoting = true;
      const done = p => { pop.style.display = 'none'; pop.onclick = null; document.removeEventListener('keydown', esc); promoting = false; resolve(p); };
      const esc = e => { if (e.key === 'Escape') done(null); };
      document.addEventListener('keydown', esc);
      pop.onclick = e => {
        const b = e.target.closest('button');
        if (b) done(b.dataset.p || null); else if (e.target === pop) done(null);
      };
    });
  }

  // Tap-to-move (phones): tap a piece to select it and see its legal squares, then tap a destination.
  let selected = null;
  const targets = sq => game.moves({ square: sq, verbose: true }).map(m => m.to);
  function deselect() {
    if (selected) $('#board .square-' + selected).removeClass('sel');
    selected = null;
  }
  function select(sq) {
    clearOverlays(); deselect();
    selected = sq;
    $('#board .square-' + sq).addClass('sel');
    for (const to of targets(sq)) {                        // legal squares, shown at once without the model
      const { x, y, size } = squareXY(to);
      const el = document.createElement('div');
      el.className = 'move-hint ' + (game.get(to) ? 'capture' : 'dot');
      Object.assign(el.style, { left: x + 'px', top: y + 'px', width: size + 'px', height: size + 'px' });
      $id('moveHints').appendChild(el); hints.push(el);
    }
    if (modelReady) showPiece(sq, true);                   // then the model's probabilities, if it is loaded
  }
  function tryMove(source, target) {
    const legal = game.moves({ square: source, verbose: true }).filter(m => m.to === target);
    if (!legal.length) { deselect(); clearOverlays(); board.position(game.fen()); return 'snapback'; }
    if (legal.some(m => m.flags.includes('p'))) {          // pawn reached the last rank: ask which piece
      askPromotion(turnColor()).then(p => (p ? humanMove(source, target, p) : (deselect(), clearOverlays(), board.position(game.fen()))));
      return;
    }
    humanMove(source, target);
  }
  function onDrop(source, target) {
    if (source === target) {                               // a tap, not a drag
      if (selected === source) { deselect(); clearOverlays(); } else select(source);
      return;
    }
    clearOverlays();
    return tryMove(source, target);
  }

  // ---- overlays: piece heatmap for the side to move, and one piece's destinations ----
  let hints = [], tinted = [];
  function clearOverlays() {
    hints.forEach(h => h.remove()); hints = [];
    tinted.forEach(sq => $('#board .square-' + sq).css('background', '')); tinted = [];
    $id('distPanel').innerHTML = ''; $id('heatmapLegend').innerHTML = '';
  }
  function squareXY(sq) {
    const size = $('#board').width() / 8, f = sq.charCodeAt(0) - 97, r = +sq[1];
    const [c, row] = board.orientation() === 'white' ? [f, 8 - r] : [7 - f, r - 1];
    return { x: c * size, y: row * size, size };
  }
  const heat = p => `rgba(255,${Math.round(255 * (1 - Math.min(p, 1)))},${Math.round(255 * (1 - Math.min(p, 1)))},.75)`;
  const table = ps => {
    const max = ps[0].p;
    return '<table>' + ps.map(m => `<tr><td>${m.san}</td><td class="prob">${(m.p * 100).toFixed(1)}%<span class="bar" style="width:${Math.max(2, Math.round(m.p / max * 80))}px"></span></td></tr>`).join('') + '</table>';
  };

  async function withScores(fn, quiet) {   // run fn(moves) only if the position and prompt did not change meanwhile
    const before = state();
    setBusy(true, 'Computing probabilities…');
    try { const moves = await scores(); modelReady = true; if (state() === before && moves && moves.length) fn(moves); }
    catch (e) { if (!quiet) setStatus('Error: ' + friendly(e)); }
    finally { setBusy(false); }
    kick();                         // an engine turn that arrived while busy
  }

  function showProbs() {
    if (busy || promoting || game.game_over()) return;
    const c = turnColor();
    withScores(moves => {
      clearOverlays();
      const ps = softmax(moves, temperature(c)).sort((a, b) => b.p - a.p);
      const bySq = {};
      for (const m of ps) bySq[m.uci.slice(0, 2)] = (bySq[m.uci.slice(0, 2)] || 0) + m.p;
      for (const [sq, p] of Object.entries(bySq)) { $('#board .square-' + sq).css('background', heat(p)); tinted.push(sq); }
      $id('heatmapLegend').innerHTML = `<div class="note">${c}, ${short(promptOf(c))}: most likely moves</div>`;
      $id('distPanel').innerHTML = table(ps.slice(0, 10));
    });
  }

  function showPiece(square, quiet) {
    const c = turnColor();
    withScores(moves => {
      clearOverlays();
      if (selected) $('#board .square-' + selected).addClass('sel');
      const mine = moves.filter(m => m.uci.startsWith(square));
      if (!mine.length) return;
      const ps = ($id('distGlobal').checked ? softmax(moves, temperature(c)).filter(m => m.uci.startsWith(square))
                                            : softmax(mine, temperature(c))).sort((a, b) => b.p - a.p);
      for (const m of ps) {
        const { x, y, size } = squareXY(m.uci.slice(2, 4));
        const el = document.createElement('div');
        el.className = 'move-hint' + (m.san.includes('x') ? ' capture' : '');
        Object.assign(el.style, { left: x + 'px', top: y + 'px', width: size + 'px', height: size + 'px', background: heat(m.p) });
        $id('moveHints').appendChild(el); hints.push(el);
      }
      $id('heatmapLegend').innerHTML = `<div class="note">${c}, ${short(promptOf(c))}: moves of the piece on ${square}</div>`;
      $id('distPanel').innerHTML = table(ps);
    }, quiet);
  }

  function updateBoxes() {
    for (const c of ['white', 'black']) $id('box_' + c).classList.toggle('to-move', c === turnColor() && !game.game_over());
  }

  // ---- controls: nothing is computed until a button or a piece is clicked ----
  function newGame() {
    if (busy || promoting) return;
    stopAutoplay();
    game.reset(); cache.clear(); moveLog.length = 0;
    syncOrientation();
    forceOnce = isEngine('white') && !isEngine('black');      // engine plays White against you: it opens
    refresh(); kick();
  }
  // Takes back your last move and the engine's reply, so it is your turn again (one ply if both sides are
  // human or both are engines).
  function undo() {
    if (busy || promoting) return;
    stopAutoplay(); forceOnce = false;
    if (!game.undo()) return;
    moveLog.pop();
    if (isEngine('white') !== isEngine('black'))
      while (isEngine(turnColor()) && game.history().length) { game.undo(); moveLog.pop(); }
    refresh();
  }
  for (const c of ['white', 'black']) {
    $id('temp_' + c).addEventListener('input', e => { $id('tempVal_' + c).textContent = e.target.value; });
    $id('prompt_' + c).addEventListener('input', clearOverlays);     // shown probabilities belonged to the old prompt
    $id('who_' + c).addEventListener('change', () => {
      if (!isEngine(c)) stopAutoplay();
      syncOrientation();
      forceOnce = isEngine(c) && turnColor() === c;                    // switched to Engine on its turn: move now
      refresh(); kick();
    });
  }
  $id('who_black').value = 'engine';
  document.querySelectorAll('[data-prompt]').forEach(b => b.addEventListener('click', () => {
    $id('prompt_' + b.dataset.color).value = b.dataset.prompt; clearOverlays();
  }));
  $id('newGameBtn').onclick = newGame;
  $id('undoBtn').onclick = undo;
  $id('engineMoveBtn').onclick = engineMove;
  $id('autoplayBtn').onclick = () => (autoplay ? stopAutoplay() : startAutoplay());
  $id('probsBtn').onclick = showProbs;
  $id('flipBtn').onclick = () => { board.flip(); deselect(); clearOverlays(); markLastMove(); };
  $('#board').on('click', '.square-55d63', function () {
    const sq = (this.className.match(/square-([a-h][1-8])/) || [])[1];
    if (!sq || busy || promoting) return;
    if (selected) {                                        // second tap of tap-to-move (empty square)
      if (targets(selected).includes(sq)) tryMove(selected, sq);
      else if (!(game.get(sq) && game.get(sq).color === game.turn())) { deselect(); clearOverlays(); }
      return;
    }
    // your own pieces are handled by the tap in onDrop; on the engine's turn a click shows its probabilities
    if (game.get(sq) && game.get(sq).color === game.turn() && (isEngine(turnColor()) || autoplay || game.game_over())) showPiece(sq);
  });

  board = Chessboard('board', {
    draggable: true, position: 'start', onDragStart, onDrop,
    pieceTheme: 'vendor/img/chesspieces/wikipedia/{piece}.png',
  });
  window.addEventListener('resize', () => { board.resize(); deselect(); clearOverlays(); markLastMove(); });
  // resume(): call once the model becomes usable, so a waiting engine turn is played.
  const resume = () => { modelReady = true; if (isEngine(turnColor()) && !isEngine(otherColor(turnColor()))) forceOnce = true; kick(); };
  return { newGame, setBusy, setStatus, clearOverlays, resume };
}
