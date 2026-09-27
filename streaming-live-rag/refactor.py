import re

with open('static/index.html', 'r', encoding='utf-8') as f:
    html = f.read()

# 1. Replace CSS
new_css = """
    :root {
      --bg-main: #ffffff;
      --bg-sidebar: #f8fafc;
      --bg-card: #ffffff;
      --bg-input: #ffffff;
      --border-color: #e2e8f0;
      --accent: #4F46E5;
      --accent-hover: #4338ca;
      --text-main: #0f172a;
      --text-muted: #475569;
      --text-dim: #64748b;
      --success: #10b981;
      --warning: #f59e0b;
      --error: #ef4444;
      --thought-bg: #f8fafc;
      --user-msg-bg: #f1f5f9;
    }
    [data-theme="dark"] {
      --bg-main: #212121;
      --bg-sidebar: #171717;
      --bg-card: #212121;
      --bg-input: #2f2f2f;
      --border-color: #333333;
      --accent: #6366f1;
      --accent-hover: #4f46e5;
      --text-main: #ececec;
      --text-muted: #b4b4b4;
      --text-dim: #888888;
      --thought-bg: #171717;
      --user-msg-bg: #2f2f2f;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background-color: var(--bg-main);
      color: var(--text-main);
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      height: 100vh;
      display: flex;
      flex-direction: column;
      overflow: hidden;
      -webkit-font-smoothing: antialiased;
      transition: background-color 0.2s, color 0.2s;
    }

    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: var(--border-color); border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--text-dim); }

    /* Header */
    header {
      height: 56px;
      background: var(--bg-main);
      border-bottom: 1px solid var(--border-color);
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 16px;
      z-index: 20;
    }
    .brand { display: flex; align-items: center; gap: 10px; }
    .brand-logo {
      width: 28px; height: 28px; border-radius: 6px;
      background: var(--accent); color: white;
      display: flex; align-items: center; justify-content: center;
    }
    .brand-title { font-weight: 600; font-size: 0.95rem; display: flex; align-items: center; gap: 8px; color: var(--text-main); }
    .brand-sub { font-size: 0.75rem; color: var(--text-muted); display: none; }
    @media (min-width: 768px) { .brand-sub { display: block; } }

    .header-actions { display: flex; align-items: center; gap: 12px; }
    .pill {
      font-size: 0.75rem; font-weight: 500; padding: 4px 10px; border-radius: 6px;
      display: inline-flex; align-items: center; gap: 6px;
      background: var(--bg-sidebar); border: 1px solid var(--border-color); color: var(--text-muted);
    }
    .pill.active { color: var(--text-main); }
    .pill-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--text-dim); }
    .pill-green.active .pill-dot { background: var(--success); }
    .pill-blue.active .pill-dot { background: var(--accent); }

    .header-btn {
      background: transparent;
      border: 1px solid transparent;
      color: var(--text-muted);
      font-size: 0.8rem;
      font-weight: 500;
      padding: 6px 8px;
      border-radius: 6px;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      transition: all 0.2s;
    }
    .header-btn:hover { background: var(--bg-sidebar); color: var(--text-main); }
    .header-btn.outline { border-color: var(--border-color); }
    .header-btn.outline:hover { border-color: var(--text-dim); }
    .header-btn.primary { background: var(--accent); color: white; border-color: var(--accent); }
    .header-btn.primary:hover { background: var(--accent-hover); color: white; }

    /* App Container */
    .app-container { display: flex; flex: 1; height: calc(100vh - 56px); position: relative; }

    /* Sidebar */
    .sidebar {
      width: 280px;
      background: var(--bg-sidebar);
      border-right: 1px solid var(--border-color);
      display: flex; flex-direction: column; overflow-y: auto; padding: 16px; gap: 20px; z-index: 10;
    }

    /* Live Toggle */
    .live-toggle-card {
      display: flex; flex-direction: column; gap: 8px;
      padding-bottom: 16px; border-bottom: 1px solid var(--border-color);
    }
    .live-toggle-row { display: flex; justify-content: space-between; align-items: center; }
    .live-toggle-title { font-size: 0.85rem; font-weight: 600; color: var(--text-main); display: flex; align-items: center; gap: 6px; }
    .live-toggle-sub { font-size: 0.75rem; color: var(--text-muted); line-height: 1.4; }

    .switch { position: relative; display: inline-block; width: 36px; height: 20px; }
    .switch input { opacity: 0; width: 0; height: 0; }
    .slider {
      position: absolute; cursor: pointer; inset: 0;
      background-color: var(--border-color); transition: .3s; border-radius: 20px;
    }
    .slider:before {
      position: absolute; content: ""; height: 14px; width: 14px; left: 3px; bottom: 3px;
      background-color: #fff; transition: .3s; border-radius: 50%;
    }
    input:checked + .slider { background-color: var(--accent); }
    input:checked + .slider:before { transform: translateX(16px); }

    /* Section Headers */
    .sidebar-section-title {
      font-size: 0.75rem; font-weight: 600; color: var(--text-main);
      display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;
    }

    /* Scenarios */
    .scenario-item {
      padding: 8px 10px; border-radius: 6px; cursor: pointer; transition: all 0.2s; margin-bottom: 2px;
      border: 1px solid transparent; display: flex; flex-direction: column; gap: 4px;
    }
    .scenario-item:hover { background: var(--bg-main); border-color: var(--border-color); }
    .scenario-item.active { background: var(--bg-main); border-color: var(--accent); }
    .scenario-top { display: flex; justify-content: space-between; align-items: center; }
    .scenario-title { font-size: 0.8rem; font-weight: 500; color: var(--text-main); display: flex; align-items: center; gap: 6px; }
    .scenario-tag { font-size: 0.65rem; font-weight: 500; padding: 2px 6px; border-radius: 4px; background: var(--bg-sidebar); border: 1px solid var(--border-color); color: var(--text-muted); }
    .scenario-query { font-size: 0.75rem; color: var(--text-muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

    /* HUD Grid */
    .hud-mini-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .hud-mini-card { padding: 8px 0; border-bottom: 1px solid var(--border-color); }
    .hud-mini-label { font-size: 0.7rem; color: var(--text-muted); font-weight: 500; margin-bottom: 2px; }
    .hud-mini-val { font-size: 0.9rem; font-weight: 600; color: var(--text-main); }

    /* Corpus */
    .corpus-mini-box { font-size: 0.75rem; }
    .corpus-chunk-badge {
      display: inline-block; background: var(--bg-main); color: var(--text-muted);
      border: 1px solid var(--border-color); border-radius: 4px;
      padding: 2px 6px; font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; margin: 2px 2px 2px 0;
    }

    /* Chat Main */
    .chat-main { flex: 1; display: flex; flex-direction: column; background: var(--bg-main); position: relative; overflow: hidden; }
    .chat-messages { flex: 1; overflow-y: auto; padding: 24px 20px 170px; scroll-behavior: smooth; }
    .chat-container { max-width: 800px; margin: 0 auto; width: 100%; display: flex; flex-direction: column; gap: 24px; }

    /* Hero */
    .hero-banner { text-align: center; padding: 60px 20px; margin-top: 40px; }
    .hero-icon { margin-bottom: 16px; display: inline-flex; align-items: center; justify-content: center; width: 48px; height: 48px; border-radius: 50%; background: var(--bg-sidebar); border: 1px solid var(--border-color); color: var(--text-main); }
    .hero-title { font-size: 1.5rem; font-weight: 600; color: var(--text-main); margin-bottom: 12px; }
    .hero-sub { font-size: 0.9rem; color: var(--text-muted); max-width: 580px; margin: 0 auto 24px; line-height: 1.5; }
    .hero-badges { display: flex; justify-content: center; gap: 8px; flex-wrap: wrap; }

    /* Chat Rows */
    .chat-row { display: flex; gap: 16px; animation: fadeIn 0.2s ease-out; }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }
    .chat-row.user { justify-content: flex-end; }
    .msg-avatar { width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center; justify-content: center; flex-shrink: 0; border: 1px solid var(--border-color); background: var(--bg-sidebar); color: var(--text-main); }
    .msg-avatar.assistant { border-color: transparent; background: transparent; }
    .msg-bubble { max-width: 80%; line-height: 1.6; }
    .chat-row.user .msg-bubble { background: var(--user-msg-bg); border-radius: 16px; padding: 12px 16px; color: var(--text-main); font-size: 0.95rem; }
    .chat-row.assistant .msg-bubble { flex: 1; max-width: 100%; color: var(--text-main); font-size: 0.95rem; }
    
    .user-msg-meta { display: none; } /* removed audio text */

    /* Reasoning UI */
    .thinking-accordion { margin-bottom: 16px; font-size: 0.85rem; }
    .thinking-header { display: flex; align-items: center; gap: 8px; cursor: pointer; user-select: none; color: var(--text-muted); font-weight: 500; transition: color 0.2s; }
    .thinking-header:hover { color: var(--text-main); }
    .thinking-status-pill { font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: var(--text-muted); margin-left: 8px; }
    .thinking-toggle-icon { width: 14px; height: 14px; transition: transform 0.2s; }
    .thinking-accordion.collapsed .thinking-toggle-icon { transform: rotate(-90deg); }
    .thinking-accordion.collapsed .thinking-body { display: none; }

    .turn-inspector-tabs { display: inline-flex; gap: 12px; margin-left: auto; margin-right: 8px; }
    .turn-tab-btn { background: none; border: none; color: var(--text-muted); font-size: 0.75rem; font-weight: 500; cursor: pointer; }
    .turn-tab-btn:hover { color: var(--text-main); }
    .turn-tab-btn.active { color: var(--text-main); font-weight: 600; }

    .thinking-body { margin-top: 12px; padding: 12px 16px; background: var(--thought-bg); border-left: 2px solid var(--border-color); border-radius: 0 8px 8px 0; max-height: 400px; overflow-y: auto; }
    .thought-stream-pane { display: block; }
    .pipeline-trace-pane { display: none; }

    .thought-stream-paragraph { margin-bottom: 12px; font-size: 0.85rem; color: var(--text-muted); }
    .thought-stream-meta { display: flex; gap: 8px; margin-bottom: 4px; font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: var(--text-dim); }
    .thought-line-break { height: 8px; }
    .thought-code { font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; background: var(--bg-main); padding: 2px 4px; border-radius: 4px; border: 1px solid var(--border-color); }
    .thought-bullet { display: flex; gap: 8px; margin-left: 8px; margin-top: 4px; }
    .bullet-dot { color: var(--text-dim); }

    /* Timeline Items */
    .thought-item { position: relative; padding-left: 24px; margin-bottom: 16px; }
    .thought-rail { position: absolute; left: 7px; top: 20px; bottom: -16px; width: 2px; background: var(--border-color); }
    .thought-item:last-child .thought-rail { display: none; }
    .thought-dot { position: absolute; left: 0; top: 2px; width: 16px; height: 16px; border-radius: 50%; background: var(--bg-sidebar); border: 2px solid var(--text-dim); display: flex; align-items: center; justify-content: center; }
    .thought-dot svg { width: 10px; height: 10px; color: var(--text-dim); }
    .thought-meta { display: flex; gap: 8px; margin-bottom: 4px; }
    .thought-time { font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: var(--text-muted); }
    .thought-badge { font-size: 0.7rem; font-weight: 500; color: var(--text-dim); background: var(--bg-main); border: 1px solid var(--border-color); padding: 1px 6px; border-radius: 4px; }
    .thought-title { font-size: 0.85rem; font-weight: 600; color: var(--text-main); margin-bottom: 4px; }
    .thought-chunk-box { margin: 8px 0; padding: 8px 12px; background: var(--bg-main); border: 1px solid var(--border-color); border-radius: 6px; }
    .thought-chunk-label { font-size: 0.7rem; font-weight: 600; color: var(--text-muted); margin-bottom: 4px; }
    .thought-chunk-text { font-size: 0.8rem; color: var(--text-main); font-family: monospace; }
    .thought-text { font-size: 0.85rem; color: var(--text-muted); }
    .thought-note { font-size: 0.8rem; color: var(--text-dim); margin-top: 4px; }

    .live-thinking-indicator { display: inline-flex; align-items: center; gap: 8px; font-size: 0.8rem; color: var(--text-muted); margin-bottom: 8px; }
    .pulse-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--text-dim); animation: pulse 1.5s infinite; }
    @keyframes pulse { 0% { opacity: 0.4; transform: scale(0.9); } 50% { opacity: 1; transform: scale(1.1); } 100% { opacity: 0.4; transform: scale(0.9); } }

    /* Answer */
    .answer-text { font-size: 0.95rem; color: var(--text-main); line-height: 1.6; }
    .answer-text p { margin-bottom: 12px; }
    .answer-text ul, .answer-text ol { margin-left: 20px; margin-bottom: 12px; }
    .citation-chip { background: var(--bg-sidebar); border: 1px solid var(--border-color); color: var(--text-muted); border-radius: 4px; padding: 1px 6px; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; cursor: pointer; transition: all 0.2s; margin: 0 2px; }
    .citation-chip:hover { border-color: var(--text-main); color: var(--text-main); }

    .verification-footer { display: flex; align-items: center; gap: 8px; margin-top: 16px; padding: 8px 12px; border-radius: 6px; background: var(--bg-sidebar); font-size: 0.8rem; color: var(--text-muted); font-weight: 500; }
    .verification-footer svg { width: 14px; height: 14px; }

    /* Input */
    .chat-input-bar { position: absolute; bottom: 24px; left: 50%; transform: translateX(-50%); width: calc(100% - 48px); max-width: 800px; display: flex; flex-direction: column; gap: 12px; z-index: 30; }
    .quick-chips-row { display: flex; gap: 8px; overflow-x: auto; padding-bottom: 4px; scrollbar-width: none; }
    .quick-chips-row::-webkit-scrollbar { display: none; }
    .quick-chip { background: var(--bg-main); border: 1px solid var(--border-color); color: var(--text-muted); font-size: 0.8rem; padding: 6px 12px; border-radius: 20px; cursor: pointer; white-space: nowrap; transition: all 0.2s; display: inline-flex; align-items: center; gap: 6px; }
    .quick-chip:hover { color: var(--text-main); background: var(--bg-sidebar); border-color: var(--text-dim); }

    .input-box-wrapper { background: var(--bg-input); border: 1px solid var(--border-color); border-radius: 24px; padding: 8px 12px; display: flex; align-items: flex-end; gap: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); transition: border-color 0.2s, box-shadow 0.2s; }
    .input-box-wrapper:focus-within { border-color: var(--text-dim); box-shadow: 0 4px 12px rgba(0,0,0,0.08); }

    .mic-action-btn { width: 32px; height: 32px; border-radius: 16px; background: transparent; border: none; color: var(--text-muted); display: flex; align-items: center; justify-content: center; cursor: pointer; flex-shrink: 0; transition: all 0.2s; }
    .mic-action-btn:hover { background: var(--bg-sidebar); color: var(--text-main); }
    .mic-action-btn.recording { background: #fee2e2; color: var(--error); animation: pulseMic 1.5s infinite; }
    @keyframes pulseMic { 0% { transform: scale(1); } 50% { transform: scale(1.1); } 100% { transform: scale(1); } }

    .chat-textarea { flex: 1; background: transparent; border: none; outline: none; color: var(--text-main); font-family: inherit; font-size: 1rem; line-height: 1.5; resize: none; max-height: 160px; padding: 4px 0; }
    .chat-textarea::placeholder { color: var(--text-dim); }

    .send-action-btn { width: 32px; height: 32px; border-radius: 16px; background: var(--accent); border: none; color: white; display: flex; align-items: center; justify-content: center; cursor: pointer; flex-shrink: 0; transition: all 0.2s; }
    .send-action-btn:hover { background: var(--accent-hover); transform: translateY(-1px); }
    .send-action-btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; background: var(--border-color); color: var(--text-dim); }

    .input-caption { text-align: center; font-size: 0.75rem; color: var(--text-dim); display: flex; align-items: center; justify-content: center; gap: 4px; }
    .input-caption svg { width: 14px; height: 14px; }

    /* Modal */
    .modal-backdrop { position: fixed; inset: 0; background: rgba(0, 0, 0, 0.4); backdrop-filter: blur(4px); z-index: 100; display: none; align-items: center; justify-content: center; padding: 20px; }
    .modal-backdrop.open { display: flex; }
    .modal-card { background: var(--bg-main); border: 1px solid var(--border-color); border-radius: 12px; max-width: 520px; width: 100%; padding: 24px; box-shadow: 0 10px 25px rgba(0,0,0,0.1); max-height: 90vh; overflow-y: auto; }
    .modal-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
    .modal-title { font-size: 1.1rem; font-weight: 600; color: var(--text-main); display: flex; align-items: center; gap: 8px; }
    .close-btn { background: none; border: none; color: var(--text-muted); font-size: 1.25rem; cursor: pointer; }
    .close-btn:hover { color: var(--text-main); }
    label.field-label { display: block; font-size: 0.8rem; color: var(--text-main); margin-bottom: 6px; font-weight: 500; }
    input.field-input, select.field-select { width: 100%; background: var(--bg-input); border: 1px solid var(--border-color); border-radius: 6px; color: var(--text-main); padding: 10px 12px; font-size: 0.9rem; margin-bottom: 16px; outline: none; transition: border-color 0.2s; }
    input.field-input:focus, select.field-select:focus { border-color: var(--accent); }
"""

html = re.sub(r'<style>.*?</style>', f'<style>\n{new_css}\n  </style>', html, flags=re.DOTALL)

# Add lucide and initialize script
lucide_script = '<script src="https://unpkg.com/lucide@latest"></script>'
html = html.replace('<style>', f'{lucide_script}\n  <style>')
# At bottom
html = html.replace('connectWs();\n  </script>', 'connectWs();\n    lucide.createIcons();\n  </script>')

# Add toggleTheme function
theme_js = """
    // ─── THEME ─────────────────────────────────────────────
    function toggleTheme() {
      const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
      document.documentElement.setAttribute('data-theme', isDark ? 'light' : 'dark');
      localStorage.setItem('theme', isDark ? 'light' : 'dark');
    }
    // init theme
    if (localStorage.getItem('theme') === 'dark' || (!localStorage.getItem('theme') && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
      document.documentElement.setAttribute('data-theme', 'dark');
    } else {
      document.documentElement.setAttribute('data-theme', 'light');
    }
"""
html = html.replace('// ─── STATE MANAGEMENT ─────────────────────────────────────────────', theme_js + '\n    // ─── STATE MANAGEMENT ─────────────────────────────────────────────')

# Header rewrite
header_html = """
  <header>
    <div class="brand">
      <div class="brand-logo"><i data-lucide="zap" width="16" height="16"></i></div>
      <div>
        <div class="brand-title">Streaming Live RAG <span class="pill">Theme 4</span></div>
        <div class="brand-sub">Samsung PRISM GenAI Hackathon</div>
      </div>
    </div>
    <div class="header-actions">
      <span class="pill pill-green active" id="apiStatusPill"><span class="pill-dot"></span> Pipeline Ready</span>
      <span class="pill pill-blue active" id="corpusPill"><span class="pill-dot"></span> 4 Chunks Indexed</span>
      <button class="header-btn" onclick="toggleTheme()" title="Toggle Theme"><i data-lucide="sun-moon" width="16" height="16"></i></button>
      <button class="header-btn outline" onclick="openByokModal()"><i data-lucide="settings" width="14" height="14"></i> Settings</button>
      <button class="header-btn outline" onclick="reseedCorpus()"><i data-lucide="refresh-cw" width="14" height="14"></i> Reseed Data</button>
    </div>
  </header>
"""
html = re.sub(r'<header>.*?</header>', header_html, html, flags=re.DOTALL)

# Sidebar rewrite
sidebar_regex = r'<div class="sidebar">.*?</div>\s+<!-- Main Chat Thread -->'
new_sidebar = """
    <div class="sidebar">

      <div class="live-toggle-card">
        <div class="live-toggle-row">
          <div class="live-toggle-title">
            <i data-lucide="radio" width="16" height="16" style="color:var(--error);"></i> Live Stream Mode
          </div>
          <label class="switch">
            <input type="checkbox" id="liveModeToggle" checked onchange="toggleLiveMode(this.checked)">
            <span class="slider"></span>
          </label>
        </div>
        <div class="live-toggle-sub">
          Evaluates speech and typing dynamically. Continues thinking without resetting.
        </div>
      </div>

      <div style="display:flex; justify-content:space-between; align-items:center;">
        <div style="font-size:0.75rem; color:var(--text-muted);">
          Session: <code id="lblSession" style="color:var(--text-main);">session_demo_01</code>
        </div>
        <button class="header-btn outline" onclick="clearSessionHistory()" style="padding:4px 8px; font-size:0.7rem;">
          <i data-lucide="plus" width="12" height="12"></i> New Chat
        </button>
      </div>

      <div>
        <div class="sidebar-section-title">
          <span style="display:flex; align-items:center; gap:6px;"><i data-lucide="bar-chart-2" width="14" height="14"></i> Session Stats</span>
        </div>
        <div class="hud-mini-grid">
          <div class="hud-mini-card">
            <div class="hud-mini-label">Latency</div>
            <div class="hud-mini-val" id="hudLatency">— ms</div>
          </div>
          <div class="hud-mini-card">
            <div class="hud-mini-label">Grounding</div>
            <div class="hud-mini-val" id="hudGrounding">—</div>
          </div>
          <div class="hud-mini-card">
            <div class="hud-mini-label">Intents</div>
            <div class="hud-mini-val" id="hudIntents">0 Intents</div>
          </div>
          <div class="hud-mini-card">
            <div class="hud-mini-label">Version</div>
            <div class="hud-mini-val" id="hudVersion">v1</div>
          </div>
        </div>
      </div>

      <div>
        <div class="sidebar-section-title">
          <span style="display:flex; align-items:center; gap:6px;"><i data-lucide="target" width="14" height="14"></i> Benchmark Scenarios</span>
        </div>
        <div class="scenario-item" onclick="loadAndRunScenario(1)">
          <div class="scenario-top">
            <span class="scenario-title"><i data-lucide="zap" width="12" height="12"></i> 1. Early Retrieval (G2)</span>
            <span class="scenario-tag">Speculative</span>
          </div>
          <div class="scenario-query">"What is the maximum capacity of the workshop hall?"</div>
        </div>
        <div class="scenario-item active" onclick="loadAndRunScenario(2)">
          <div class="scenario-top">
            <span class="scenario-title"><i data-lucide="shuffle" width="12" height="12"></i> 2. Multi-Intent (G3)</span>
            <span class="scenario-tag">3 Beams</span>
          </div>
          <div class="scenario-query">"What is the venue capacity, what is the cancellation penalty..."</div>
        </div>
        <div class="scenario-item" onclick="loadAndRunScenario(3)">
          <div class="scenario-top">
            <span class="scenario-title"><i data-lucide="message-square" width="12" height="12"></i> 3. Chit-Chat (C2)</span>
            <span class="scenario-tag">0 Retrieval</span>
          </div>
          <div class="scenario-query">"Hello! Can you help me today?"</div>
        </div>
        <div class="scenario-item" onclick="loadAndRunScenario(4)">
          <div class="scenario-top">
            <span class="scenario-title"><i data-lucide="list" width="12" height="12"></i> 4. Reformat (C4)</span>
            <span class="scenario-tag">Cache</span>
          </div>
          <div class="scenario-query">"Can you format that in two bullet points?"</div>
        </div>
        <div class="scenario-item" onclick="loadAndRunScenario(5)">
          <div class="scenario-top">
            <span class="scenario-title"><i data-lucide="edit-3" width="12" height="12"></i> 5. Refinement (C7)</span>
            <span class="scenario-tag">v2</span>
          </div>
          <div class="scenario-query">"Actually, what if there are 50 attendees?"</div>
        </div>
        <div class="scenario-item" onclick="loadAndRunScenario(6)">
          <div class="scenario-top">
            <span class="scenario-title"><i data-lucide="shield" width="12" height="12"></i> 6. Abstention (C6)</span>
            <span class="scenario-tag">Guard</span>
          </div>
          <div class="scenario-query">"What is the domestic train meal allowance in Mumbai?"</div>
        </div>
      </div>

      <div>
        <div class="sidebar-section-title">
          <span style="display:flex; align-items:center; gap:6px;"><i data-lucide="folder" width="14" height="14"></i> Indexed Corpus</span>
          <button onclick="refreshCorpus()" style="background:none; border:none; color:var(--text-muted); cursor:pointer;"><i data-lucide="refresh-cw" width="12" height="12"></i></button>
        </div>
        <div class="corpus-mini-box">
          <div id="corpusSummaryText" style="margin-bottom:6px; color:var(--text-muted);">Loading corpus data...</div>
          <div id="corpusDocsList"></div>
          <div style="margin-top:10px; border-top:1px solid var(--border-color); padding-top:8px;">
            <input type="file" id="docUploadInput" style="display:none;" onchange="handleFileUpload(event)" accept=".pdf,.docx,.pptx,.xlsx,.csv,.tsv,.json,.yaml,.yml,.html,.htm,.txt,.md" />
            <button class="header-btn outline" onclick="document.getElementById('docUploadInput').click()" style="width:100%; justify-content:center; padding:6px;">
              <i data-lucide="upload" width="14" height="14"></i> Upload Custom File
            </button>
          </div>
        </div>
      </div>
    </div>
    <!-- Main Chat Thread -->"""
html = re.sub(sidebar_regex, new_sidebar, html, flags=re.DOTALL)

# Hero Banner inside Main Chat
hero_regex = r'<div class="hero-banner" id="heroBanner">.*?</div>'
new_hero = """<div class="hero-banner" id="heroBanner">
            <div class="hero-icon"><i data-lucide="zap" width="24" height="24"></i></div>
            <div class="hero-title">Streaming Live Speech RAG</div>
            <div class="hero-sub">
              Experience real-time speculative search with uninterrupted continuous thought. Speak into the microphone or test a benchmark scenario below.
            </div>
            <div class="hero-badges">
              <span class="pill">Speculative Pre-Retrieval</span>
              <span class="pill">Continuous Thought</span>
              <span class="pill">100% Grounded</span>
            </div>
          </div>"""
html = re.sub(hero_regex, new_hero, html, flags=re.DOTALL)

# Input Bar rewrite
input_bar_regex = r'<div class="chat-input-bar">.*?</div>\s+</div>\s+</div>\s+<!-- Modal'
new_input = """<div class="chat-input-bar">
        <div class="quick-chips-row">
          <div class="quick-chip" onclick="loadAndRunScenario(2)"><i data-lucide="shuffle" width="14" height="14"></i> Multi-Intent</div>
          <div class="quick-chip" onclick="loadAndRunScenario(1)"><i data-lucide="zap" width="14" height="14"></i> Early Retrieval</div>
          <div class="quick-chip" onclick="loadAndRunScenario(3)"><i data-lucide="message-square" width="14" height="14"></i> Chit-Chat</div>
          <div class="quick-chip" onclick="loadAndRunScenario(4)"><i data-lucide="list" width="14" height="14"></i> Reformat</div>
          <div class="quick-chip" onclick="loadAndRunScenario(5)"><i data-lucide="edit-3" width="14" height="14"></i> Refinement</div>
          <div class="quick-chip" onclick="loadAndRunScenario(6)"><i data-lucide="shield" width="14" height="14"></i> Abstention</div>
        </div>

        <div class="input-box-wrapper">
          <button class="mic-action-btn" id="micBtn" title="Speak into Microphone" onclick="toggleSpeechRecognition()">
            <i data-lucide="mic" width="18" height="18"></i>
          </button>
          <textarea id="chatInput" class="chat-textarea" rows="1" placeholder="Ask a question or speak into microphone..." onkeydown="handleInputKey(event)"></textarea>
          <button class="send-action-btn" id="sendBtn" title="Send message" onclick="submitUserMessage()">
            <i data-lucide="arrow-up" width="18" height="18"></i>
          </button>
        </div>

        <div class="input-caption" id="inputCaption">
          <i data-lucide="radio" width="12" height="12"></i> <strong>Live Stream ON:</strong> Evaluates speech dynamically.
        </div>
      </div>
    </div>
  </div>
  <!-- Modal"""
html = re.sub(input_bar_regex, new_input, html, flags=re.DOTALL)

# Modal Rewrite
modal_regex = r'<div class="modal-backdrop" id="byokModal">.*?</div>\s+</div>\s+<script>'
new_modal = """<div class="modal-backdrop" id="byokModal">
    <div class="modal-card">
      <div class="modal-header">
        <div class="modal-title"><i data-lucide="settings" width="18" height="18"></i> LLM Provider Settings (BYOK)</div>
        <button class="close-btn" onclick="closeByokModal()"><i data-lucide="x" width="20" height="20"></i></button>
      </div>

      <div style="font-size:0.85rem; color:var(--text-muted); margin-bottom:20px;">
        Configure third-party providers. Keys are write-only.
      </div>

      <div style="margin-bottom:20px; padding-bottom:16px; border-bottom:1px solid var(--border-color);">
        <label class="field-label">Fast Controller / Decomposer Provider</label>
        <select id="mFastProvider" class="field-select">
          <option value="groq">Groq (Default Cloud Fast)</option>
          <option value="ollama">Ollama (Local / Offline)</option>
          <option value="nvidia">NVIDIA NIM</option>
          <option value="openai">OpenAI</option>
        </select>
        <label class="field-label">Fast Model</label>
        <input type="text" id="mFastModel" class="field-input" value="openai/gpt-oss-20b" />
        <label class="field-label">Fast API Key</label>
        <input type="password" id="mFastApiKey" class="field-input" placeholder="Enter new key to update..." />
      </div>

      <div style="margin-bottom:20px;">
        <label class="field-label">Quality Synthesis Provider</label>
        <select id="mSynthProvider" class="field-select">
          <option value="gemini">Google Gemini (Default)</option>
          <option value="nvidia">NVIDIA NIM</option>
          <option value="groq">Groq</option>
          <option value="ollama">Ollama</option>
          <option value="openai">OpenAI</option>
        </select>
        <label class="field-label">Synthesis Model</label>
        <input type="text" id="mSynthModel" class="field-input" value="gemini-3.5-flash-lite" />
        <label class="field-label">Synthesis API Key</label>
        <input type="password" id="mSynthApiKey" class="field-input" placeholder="Enter new key to update..." />
      </div>

      <div id="modalStatus" style="display:none; font-size:0.85rem; padding:10px; border-radius:6px; margin-bottom:16px;"></div>

      <div style="display:flex; justify-content:flex-end; gap:12px;">
        <button class="header-btn outline" onclick="testModalProviders()">Test Connection</button>
        <button class="header-btn primary" onclick="saveModalConfig()">Save &amp; Apply</button>
      </div>
    </div>
  </div>
  <script>"""
html = re.sub(modal_regex, new_modal, html, flags=re.DOTALL)

# JS replacements for DOM injection strings
html = html.replace(
"""        <div class="msg-avatar user">👤</div>""",
"""        <div class="msg-avatar user"><i data-lucide="user" width="16" height="16"></i></div>"""
)
html = html.replace(
"""<div class="msg-avatar assistant">⚡</div>""",
"""<div class="msg-avatar assistant"><i data-lucide="cpu" width="18" height="18"></i></div>"""
)
html = html.replace(
"""<span class="thinking-brain-icon">🧠</span>""",
"""<span class="thinking-brain-icon"><i data-lucide="activity" width="14" height="14"></i></span>"""
)
html = html.replace(
"""<div class="thinking-toggle-icon">▼</div>""",
"""<div class="thinking-toggle-icon"><i data-lucide="chevron-down" width="14" height="14"></i></div>"""
)

# add lucide re-render in ensureTurnUI
html = html.replace(
"""      return ctx;
    }""",
"""      setTimeout(() => lucide.createIcons(), 0);
      return ctx;
    }""")
    
# update addThoughtItem
html = html.replace(
"""      ctx.traceList.appendChild(div);
      scrollToBottom();
    }""",
"""      ctx.traceList.appendChild(div);
      setTimeout(() => lucide.createIcons(), 0);
      scrollToBottom();
    }""")

with open('static/index.html', 'w', encoding='utf-8') as f:
    f.write(html)
