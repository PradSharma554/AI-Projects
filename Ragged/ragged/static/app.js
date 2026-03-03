// Chat functionality
const chatForm = document.getElementById('chat-form');
const messagesDiv = document.getElementById('messages');
const questionInput = document.getElementById('question');
const sendBtn = document.getElementById('send-btn');

if (chatForm) {
  chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const question = questionInput.value.trim();
    if (!question) return;

    // Add user message
    addMessage(question, 'user');
    questionInput.value = '';
    sendBtn.disabled = true;

    // Add loading message
    const loadingEl = addMessage('Thinking...', 'loading');

    try {
      const resp = await fetch('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question }),
      });
      const data = await resp.json();

      loadingEl.remove();

      if (resp.ok) {
        addAnswerMessage(data);
      } else {
        addMessage(`Error: ${data.detail || 'Something went wrong'}`, 'system');
      }
    } catch (err) {
      loadingEl.remove();
      addMessage(`Error: ${err.message}`, 'system');
    }

    sendBtn.disabled = false;
    questionInput.focus();
  });
}

function addMessage(text, type) {
  const div = document.createElement('div');
  div.className = `message ${type}`;
  div.textContent = text;
  messagesDiv.appendChild(div);
  messagesDiv.scrollTop = messagesDiv.scrollHeight;
  return div;
}

function addAnswerMessage(data) {
  const div = document.createElement('div');
  div.className = 'message assistant';
  div.textContent = data.text;

  if (data.sources && data.sources.length > 0) {
    const panel = document.createElement('details');
    panel.className = 'sources-panel';
    const summary = document.createElement('summary');
    summary.textContent = `${data.sources.length} source(s)`;
    panel.appendChild(summary);

    data.sources.forEach((src) => {
      const item = document.createElement('div');
      item.className = 'source-item';
      const section = src.section ? ` | ${src.section}` : '';
      const page = src.page ? ` | page ${src.page}` : '';
      item.innerHTML = `<div class="source-label">[${src.index}] ${src.source}${section}${page}</div>
        <div class="source-text">${src.text || ''}</div>`;
      panel.appendChild(item);
    });

    div.appendChild(panel);
  }

  messagesDiv.appendChild(div);
  messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

// Collections functionality
const ingestForm = document.getElementById('ingest-form');
const collectionsData = document.getElementById('collections-data');

if (ingestForm) {
  loadCollections();

  ingestForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const dir = document.getElementById('ingest-dir').value.trim();
    const collection = document.getElementById('ingest-collection').value.trim() || 'default';
    const status = document.getElementById('ingest-status');

    status.textContent = 'Ingesting...';

    try {
      const resp = await fetch('/api/ingest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ directory: dir, collection }),
      });
      const data = await resp.json();

      if (resp.ok) {
        status.textContent = `Done: ${data.files_processed} files, ${data.chunks_created} chunks created, ${data.files_skipped} skipped`;
        loadCollections();
      } else {
        status.textContent = `Error: ${data.detail}`;
      }
    } catch (err) {
      status.textContent = `Error: ${err.message}`;
    }
  });
}

async function loadCollections() {
  if (!collectionsData) return;

  try {
    const resp = await fetch('/api/collections');
    const data = await resp.json();

    if (data.length === 0) {
      collectionsData.textContent = 'No collections yet. Ingest some documents to get started.';
      return;
    }

    collectionsData.innerHTML = '';
    data.forEach((col) => {
      const card = document.createElement('div');
      card.className = 'col-card';
      card.innerHTML = `
        <div>
          <span class="col-name">${col.name}</span>
          <span class="col-count">${col.chunk_count} chunks</span>
        </div>
        <button class="delete-btn" onclick="deleteCollection('${col.name}')">Delete</button>`;
      collectionsData.appendChild(card);
    });
  } catch (err) {
    collectionsData.textContent = `Error loading collections: ${err.message}`;
  }
}

async function deleteCollection(name) {
  if (!confirm(`Delete collection "${name}"?`)) return;

  try {
    await fetch(`/api/collections/${name}`, { method: 'DELETE' });
    loadCollections();
  } catch (err) {
    alert(`Error: ${err.message}`);
  }
}
