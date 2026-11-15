// ClimateKG-RAG Frontend Application Core
let appState = {
  activeTab: 'overview',
  stats: null,
  graphData: null,
  network: null,
  selectedNodeId: null,
  activeFilterTypes: new Set(),
  nodeColorMap: {},
  benchmarkData: null,
  barChart: null,
  radarChart: null
};

// -----------------------------------------------------------------------------
// Pipeline Stage Explanations
// -----------------------------------------------------------------------------
const PIPELINE_STAGES = {
  1: {
    title: "Stage 1: Multi-Format Document Ingestion & Deduplication",
    desc: "The ingestion engine parses PDF, DOCX, and TXT research publications using PyMuPDF and python-docx. Text is cleaned of noise while strictly preserving sentence boundaries to avoid splitting scientific assertions across chunks. Each chunk is assigned a unique Document ID, page number, and SHA-256 content hash to detect and prevent duplicate content across corpora."
  },
  2: {
    title: "Stage 2: Climate Knowledge Graph Construction",
    desc: "Entities and relationships are extracted into a standardized climate ontology featuring 10 Node Types (ClimatePhenomenon, GreenhouseGas, Region, ClimateImpact, MitigationStrategy, AdaptationStrategy, Policy, ResearchPaper, Organization, Dataset) and 8 Relation Types (CAUSES, AFFECTS, OCCURS_IN, MITIGATED_BY, ADAPTED_BY, RECOMMENDED_BY, SUPPORTED_BY, RELATED_TO). Every edge explicitly retains the exact sentence evidence excerpt and page provenance."
  },
  3: {
    title: "Stage 3: Tri-Index Storage & Caching",
    desc: "Documents and chunks are stored across three complementary indices: (1) Scikit-learn TF-IDF matrix with sublinear term-frequency scaling, (2) Robertson Okapi BM25 lexical index, and (3) Sentence Transformers (all-MiniLM-L6-v2) 384-dimensional dense semantic embeddings precomputed and cached directly to disk in data/vector_cache/embeddings.npz for instant offline execution."
  },
  4: {
    title: "Stage 4: Reciprocal Rank Fusion (RRF) Hybrid Retrieval",
    desc: "Queries are executed in parallel across BM25 lexical search, Dense semantic similarity, and Knowledge Graph multi-hop path traversal. Results are synthesized via Reciprocal Rank Fusion: Score(d) = &Sigma; (w_m / (60 + r_m(d))), with weight bonuses for structural graph bridging. Redundant chunks are deduplicated while preserving unified graph metadata."
  },
  5: {
    title: "Stage 5: Evidence-Grounded Synthesis & Citation Verification",
    desc: "Retrieved passages and multi-hop causal pathways are passed to the Grounded Synthesis Engine. Every factual claim in the response is strictly constrained to retrieved evidence and tagged with clickable source citations [Filename, Page X, Chunk Y]. If no supporting evidence exists in the corpus, the engine returns an explicit Insufficient Evidence warning rather than hallucinating."
  }
};

// -----------------------------------------------------------------------------
// Initialization
// -----------------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", () => {
  initUI();
  loadStats();
  setupEventListeners();
});

function initUI() {
  // Method radio card highlighting
  document.querySelectorAll('input[name="retrievalMethod"]').forEach(radio => {
    radio.addEventListener('change', () => {
      document.querySelectorAll('.method-card').forEach(card => card.classList.remove('active'));
      radio.closest('.method-card').classList.add('active');
    });
  });

  // Slider listeners
  const topKSlider = document.getElementById('topKSlider');
  if (topKSlider) {
    topKSlider.addEventListener('input', (e) => {
      document.getElementById('topKValue').innerText = e.target.value;
    });
  }

  const uploadChunkSize = document.getElementById('uploadChunkSize');
  if (uploadChunkSize) {
    uploadChunkSize.addEventListener('input', (e) => {
      document.getElementById('uploadChunkSizeVal').innerText = e.target.value;
    });
  }

  const uploadChunkOverlap = document.getElementById('uploadChunkOverlap');
  if (uploadChunkOverlap) {
    uploadChunkOverlap.addEventListener('input', (e) => {
      document.getElementById('uploadChunkOverlapVal').innerText = e.target.value;
    });
  }

  const graphMaxNodes = document.getElementById('graphMaxNodes');
  if (graphMaxNodes) {
    graphMaxNodes.addEventListener('input', (e) => {
      document.getElementById('graphMaxNodesVal').innerText = e.target.value;
    });
  }
}

function setupEventListeners() {
  const qaInput = document.getElementById('qaInput');
  if (qaInput) {
    qaInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') executeQA();
    });
  }

  const compareInput = document.getElementById('compareInput');
  if (compareInput) {
    compareInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') runComparison();
    });
  }

  const graphSearchInput = document.getElementById('graphSearchInput');
  if (graphSearchInput) {
    graphSearchInput.addEventListener('input', debounce(() => {
      filterGraphSearch(graphSearchInput.value);
    }, 250));
  }
}

// -----------------------------------------------------------------------------
// Navigation
// -----------------------------------------------------------------------------
function switchTab(tabId) {
  appState.activeTab = tabId;

  document.querySelectorAll('.nav-tab').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.tab === tabId);
  });

  document.querySelectorAll('.tab-pane').forEach(pane => {
    pane.classList.toggle('active', pane.id === `tab-${tabId}`);
  });

  if (tabId === 'graph') {
    if (!appState.network) loadGraphData();
    else setTimeout(() => fitGraphView(), 200);
  } else if (tabId === 'benchmark') {
    loadBenchmarkSummary();
    loadBenchmarkQueries();
  } else if (tabId === 'documents') {
    loadDocuments();
  } else if (tabId === 'diagnostics') {
    loadDiagnostics();
  }

  lucide.createIcons();
}

function showPipelineDetail(stageNum) {
  document.querySelectorAll('.pipeline-step').forEach((el, idx) => {
    el.classList.toggle('active', idx + 1 === stageNum);
  });
  const detail = PIPELINE_STAGES[stageNum] || PIPELINE_STAGES[1];
  document.getElementById('pipelineDetailTitle').innerText = detail.title;
  document.getElementById('pipelineDetailDesc').innerText = detail.desc;
}

// -----------------------------------------------------------------------------
// Telemetry & Stats
// -----------------------------------------------------------------------------
async function loadStats() {
  try {
    const res = await fetch('/api/stats');
    const data = await res.json();
    appState.stats = data;
    appState.nodeColorMap = data.node_color_map || {};

    // Top Header Badges
    document.getElementById('topDocsCount').innerText = data.documents;
    document.getElementById('topChunksCount').innerText = data.chunks;
    document.getElementById('topEntitiesCount').innerText = data.nodes;
    document.getElementById('topEdgesCount').innerText = data.edges;

    // Overview Tab Stats
    const statDoc = document.getElementById('statDocCount');
    if (statDoc) statDoc.innerText = data.documents;
    const statChunk = document.getElementById('statChunkCount');
    if (statChunk) statChunk.innerText = data.chunks;
    const statNode = document.getElementById('statNodeCount');
    if (statNode) statNode.innerText = data.nodes;
    const statEdge = document.getElementById('statEdgeCount');
    if (statEdge) statEdge.innerText = data.edges;

    // Neo4j Badge
    const neoBadge = document.getElementById('neo4jStatusBadge');
    if (neoBadge) {
      if (data.neo4j_connected) {
        neoBadge.innerHTML = `<span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span><span>Neo4j Bolt Active</span>`;
      } else {
        neoBadge.innerHTML = `<span class="w-2 h-2 rounded-full bg-emerald-500"></span><span>NetworkX MultiDiGraph (Ready)</span>`;
      }
    }

    renderNodeTypePills(data.supported_node_types || []);
  } catch (err) {
    console.error("Failed to load telemetry stats:", err);
  }
}

// -----------------------------------------------------------------------------
// TAB 2: CLIMATE Q&A STUDIO
// -----------------------------------------------------------------------------
function setQuery(text) {
  const input = document.getElementById('qaInput');
  input.value = text;
  input.focus();
}

async function executeQA() {
  const query = document.getElementById('qaInput').value.trim();
  if (!query) {
    showToast("Please enter a climate research question.");
    return;
  }

  const selectedMethod = document.querySelector('input[name="retrievalMethod"]:checked').value;
  const topK = parseInt(document.getElementById('topKSlider').value, 10);
  const btn = document.getElementById('qaSubmitBtn');

  btn.disabled = true;
  btn.innerHTML = `<i data-lucide="loader-2" class="w-4 h-4 animate-spin"></i><span>Synthesizing Answer...</span>`;
  lucide.createIcons();

  try {
    const res = await fetch('/api/qa', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, method: selectedMethod, top_k: topK })
    });

    const data = await res.json();
    renderQAResults(data);
  } catch (err) {
    console.error("QA execution failed:", err);
    showToast("Error synthesizing answer. Check server logs.");
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<i data-lucide="send" class="w-4 h-4"></i><span>Synthesize Grounded Answer</span>`;
    lucide.createIcons();
  }
}

function renderQAResults(data) {
  const container = document.getElementById('qaResultsContainer');
  container.classList.remove('hidden');

  // Metadata
  document.getElementById('qaMetaEngine').innerText = `Synthesized by ${data.generator_type} | Method: ${data.method}`;
  
  // Confidence Badge
  const confBadge = document.getElementById('qaConfidenceBadge');
  confBadge.innerText = `Confidence: ${data.confidence}`;
  confBadge.className = `px-2.5 py-1 rounded-full font-semibold ${
    data.confidence === 'High' ? 'bg-emerald-900/60 text-emerald-400 border border-emerald-700/50' :
    data.confidence === 'Medium' ? 'bg-amber-900/60 text-amber-400 border border-amber-700/50' :
    'bg-rose-900/60 text-rose-400 border border-rose-700/50'
  }`;

  // Latency Badge
  document.getElementById('qaLatencyBadge').innerText = `⏱️ ${data.total_latency_ms} ms (Retrieval: ${data.retrieval_latency_ms} ms, QA: ${data.qa_latency_ms} ms)`;

  // Answer Text (Render Markdown with Marked.js)
  let rawAnswer = data.answer || "No response generated.";
  // Replace citation markers [1], [2] with interactive badges
  rawAnswer = rawAnswer.replace(/\[(\d+)\]/g, (match, num) => {
    return `<span class="citation-badge" onclick="highlightCitation(${num})">[${num}]</span>`;
  });
  document.getElementById('qaAnswerText').innerHTML = marked.parse(rawAnswer);

  // Graph Pathways
  const pathsSection = document.getElementById('qaGraphPathsSection');
  const pathsList = document.getElementById('qaGraphPathsList');
  if (data.graph_paths && data.graph_paths.length > 0) {
    pathsSection.classList.remove('hidden');
    pathsList.innerHTML = data.graph_paths.map(p => {
      return `<div class="px-3 py-1.5 rounded-lg bg-slate-900 border border-emerald-600/30 text-emerald-300 font-mono text-xs flex items-center space-x-1.5 shadow-sm">
        <i data-lucide="git-commit" class="w-3.5 h-3.5 text-emerald-400"></i>
        <span>${escapeHtml(p)}</span>
      </div>`;
    }).join('');
  } else {
    pathsSection.classList.add('hidden');
  }

  // Citations List
  const citList = document.getElementById('qaCitationsList');
  if (data.citations && data.citations.length > 0) {
    citList.innerHTML = data.citations.map(c => `
      <div id="citation-card-${c.source_index}" class="p-4 rounded-xl bg-slate-900/80 border border-slate-800 space-y-2 transition duration-200">
        <div class="flex items-center justify-between text-xs">
          <div class="flex items-center space-x-2">
            <span class="w-5 h-5 rounded-full bg-blue-500/20 text-blue-400 flex items-center justify-center font-bold font-mono text-[11px]">${c.source_index}</span>
            <span class="font-semibold text-white">${escapeHtml(c.doc_title)}</span>
          </div>
          <span class="font-mono text-slate-400 text-[11px]">${escapeHtml(c.filename)} (P.${c.page_number}, Chunk: ${c.chunk_id})</span>
        </div>
        <p class="text-xs text-slate-300 italic border-l-2 border-blue-500/50 pl-3 py-1">"${escapeHtml(c.snippet)}"</p>
      </div>
    `).join('');
  } else {
    citList.innerHTML = `<div class="text-xs text-slate-400">No explicit citations returned.</div>`;
  }

  // Ranked Results Passages
  const rankedList = document.getElementById('qaRankedResultsList');
  if (data.results && data.results.length > 0) {
    rankedList.innerHTML = data.results.map((r, idx) => `
      <details class="p-3.5 rounded-xl bg-slate-900/50 border border-slate-800 text-xs">
        <summary class="cursor-pointer font-medium text-slate-300 flex items-center justify-between">
          <div class="flex items-center space-x-2">
            <span class="text-emerald-400 font-bold font-mono">Rank ${r.rank || idx + 1}</span>
            <span class="text-slate-400">&bull;</span>
            <span class="text-white">${escapeHtml(r.filename)} (Page ${r.page_number})</span>
          </div>
          <span class="font-mono text-slate-400 text-[11px]">Score: ${typeof r.score === 'number' ? r.score.toFixed(4) : r.score}</span>
        </summary>
        <p class="mt-3 text-slate-300 font-mono text-[11px] leading-relaxed bg-slate-950/60 p-3 rounded-lg border border-slate-800/80">${escapeHtml(r.text)}</p>
      </details>
    `).join('');
  } else {
    rankedList.innerHTML = `<div class="text-xs text-slate-400">No passages retrieved.</div>`;
  }

  lucide.createIcons();
  container.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function highlightCitation(index) {
  const card = document.getElementById(`citation-card-${index}`);
  if (card) {
    card.scrollIntoView({ behavior: 'smooth', block: 'center' });
    card.classList.add('border-blue-500', 'bg-blue-950/30');
    setTimeout(() => {
      card.classList.remove('border-blue-500', 'bg-blue-950/30');
    }, 2000);
  }
}

// -----------------------------------------------------------------------------
// TAB 3: KNOWLEDGE GRAPH EXPLORER
// -----------------------------------------------------------------------------
function renderNodeTypePills(nodeTypes) {
  const container = document.getElementById('nodeTypePills');
  if (!container) return;

  container.innerHTML = nodeTypes.map(type => {
    const color = appState.nodeColorMap[type] || '#78909c';
    return `
      <button onclick="toggleNodeTypeFilter('${type}', this)" class="type-pill px-3 py-1 rounded-full text-xs font-semibold border flex items-center space-x-1.5 transition active" style="border-color:${color}80; background-color:${color}20; color:${color};">
        <span class="w-2 h-2 rounded-full" style="background-color:${color};"></span>
        <span>${type}</span>
      </button>
    `;
  }).join('');
}

function toggleNodeTypeFilter(type, btn) {
  if (appState.activeFilterTypes.has(type)) {
    appState.activeFilterTypes.delete(type);
    btn.style.opacity = '0.4';
    btn.classList.remove('active');
  } else {
    appState.activeFilterTypes.add(type);
    btn.style.opacity = '1.0';
    btn.classList.add('active');
  }
  loadGraphData();
}

async function loadGraphData() {
  const overlay = document.getElementById('graphLoadingOverlay');
  if (overlay) overlay.classList.remove('hidden');

  try {
    const maxNodes = document.getElementById('graphMaxNodes').value;
    const searchVal = document.getElementById('graphSearchInput').value.trim();

    let url = `/api/graph?max_nodes=${maxNodes}`;
    if (searchVal) url += `&search=${encodeURIComponent(searchVal)}`;
    if (appState.activeFilterTypes.size > 0) {
      appState.activeFilterTypes.forEach(t => {
        url += `&node_types=${encodeURIComponent(t)}`;
      });
    }

    const res = await fetch(url);
    const data = await res.json();
    appState.graphData = data;

    renderVisNetwork(data);
    populatePathSelects(data.nodes);
  } catch (err) {
    console.error("Failed to load graph data:", err);
    showToast("Error loading Knowledge Graph data.");
  } finally {
    if (overlay) overlay.classList.add('hidden');
  }
}

function renderVisNetwork(graphData) {
  const container = document.getElementById('graphNetworkCanvas');
  if (!container) return;

  const visNodes = new vis.DataSet(graphData.nodes.map(n => ({
    id: n.id,
    label: n.label,
    title: `<b>${n.label}</b><br/>Type: ${n.type}<br/>Degree: ${n.degree}`,
    color: {
      background: n.color,
      border: '#ffffff',
      highlight: { background: '#10b981', border: '#ffffff' },
      hover: { background: n.color, border: '#34d399' }
    },
    font: { color: '#f1f5f9', face: 'Inter', size: 12 },
    shape: 'dot',
    size: n.size
  })));

  const visEdges = new vis.DataSet(graphData.edges.map(e => ({
    id: e.id,
    from: e.from,
    to: e.to,
    label: e.label,
    title: `<b>${e.label}</b><br/>Evidence: <i>"${e.evidence_text}"</i><br/>Source: ${e.filename} (P.${e.page_number})`,
    arrows: { to: { enabled: true, scaleFactor: 0.7 } },
    color: { color: '#475569', highlight: '#10b981', hover: '#34d399' },
    font: { color: '#94a3b8', face: 'JetBrains Mono', size: 9, strokeWidth: 0 },
    smooth: { type: 'continuous', roundness: 0.2 }
  })));

  const options = {
    physics: {
      stabilization: { iterations: 100 },
      barnesHut: {
        gravitationalConstant: -2200,
        centralGravity: 0.25,
        springLength: 95,
        springConstant: 0.04
      }
    },
    interaction: {
      hover: true,
      tooltipDelay: 150,
      zoomView: true,
      dragView: true
    }
  };

  if (appState.network) appState.network.destroy();
  appState.network = new vis.Network(container, { nodes: visNodes, edges: visEdges }, options);

  // Click Events
  appState.network.on('click', (params) => {
    if (params.nodes.length > 0) {
      inspectNode(params.nodes[0]);
    } else if (params.edges.length > 0) {
      inspectEdge(params.edges[0]);
    } else {
      resetInspector();
    }
  });
}

function inspectNode(nodeId) {
  appState.selectedNodeId = nodeId;
  const node = appState.graphData.nodes.find(n => n.id === nodeId);
  if (!node) return;

  document.getElementById('inspectorEmptyState').classList.add('hidden');
  document.getElementById('inspectorEdgeDetails').classList.add('hidden');
  const details = document.getElementById('inspectorNodeDetails');
  details.classList.remove('hidden');

  document.getElementById('insNodeType').innerText = node.type;
  document.getElementById('insNodeType').style.color = node.color;
  document.getElementById('insNodeName').innerText = node.label;
  document.getElementById('insNodeCanonical').innerText = node.id;
  document.getElementById('insNodeDegree').innerText = node.degree;
}

function inspectEdge(edgeId) {
  const edge = appState.graphData.edges.find(e => e.id === edgeId);
  if (!edge) return;

  document.getElementById('inspectorEmptyState').classList.add('hidden');
  document.getElementById('inspectorNodeDetails').classList.add('hidden');
  const details = document.getElementById('inspectorEdgeDetails');
  details.classList.remove('hidden');

  const sourceNode = appState.graphData.nodes.find(n => n.id === edge.from);
  const targetNode = appState.graphData.nodes.find(n => n.id === edge.to);

  document.getElementById('insEdgeType').innerText = edge.label;
  document.getElementById('insEdgeLabel').innerText = `${sourceNode?.label || edge.from} ➔ ${targetNode?.label || edge.to}`;
  document.getElementById('insEdgeQuote').innerText = `"${edge.evidence_text || 'Verified domain association'}"`;
  document.getElementById('insEdgeDoc').innerText = edge.filename || 'corpus.txt';
  document.getElementById('insEdgePage').innerText = edge.page_number || 1;
  document.getElementById('insEdgeChunk').innerText = edge.source_chunk_id || 'chk_001';
}

function resetInspector() {
  document.getElementById('inspectorEmptyState').classList.remove('hidden');
  document.getElementById('inspectorNodeDetails').classList.add('hidden');
  document.getElementById('inspectorEdgeDetails').classList.add('hidden');
}

function resetGraphPhysics() {
  if (appState.network) {
    appState.network.stabilize();
    appState.network.fit({ animation: true });
  }
}

function fitGraphView() {
  if (appState.network) appState.network.fit({ animation: true });
}

function filterGraphSearch(query) {
  if (!appState.network || !appState.graphData) return;
  const qLower = query.toLowerCase().trim();
  if (!qLower) return;

  const matched = appState.graphData.nodes.find(n => n.label.toLowerCase().includes(qLower));
  if (matched) {
    appState.network.focus(matched.id, {
      scale: 1.2,
      animation: { duration: 600, easingFunction: 'easeInOutQuad' }
    });
    inspectNode(matched.id);
  }
}

async function traverseFromSelectedNode() {
  if (!appState.selectedNodeId) return;
  try {
    const res = await fetch(`/api/graph/traverse?node_id=${encodeURIComponent(appState.selectedNodeId)}&hops=2`);
    const data = await res.json();
    showToast(`Discovered ${data.total_paths} multi-hop relational pathways from ${data.node_name}`);
  } catch (err) {
    console.error("Traversal error:", err);
  }
}

// Causal Path Finder
function populatePathSelects(nodes) {
  const startSelect = document.getElementById('pathStartSelect');
  const endSelect = document.getElementById('pathEndSelect');
  if (!startSelect || !endSelect) return;

  const sorted = [...nodes].sort((a, b) => a.label.localeCompare(b.label));
  const optionsHtml = sorted.map(n => `<option value="${n.id}">${escapeHtml(n.label)} (${n.type})</option>`).join('');
  
  startSelect.innerHTML = optionsHtml;
  endSelect.innerHTML = optionsHtml;
  if (sorted.length > 1) endSelect.selectedIndex = 1;
}

async function findCausalPath() {
  const sourceId = document.getElementById('pathStartSelect').value;
  const targetId = document.getElementById('pathEndSelect').value;
  const output = document.getElementById('causalPathOutput');

  if (sourceId === targetId) {
    showToast("Please select two distinct entities.");
    return;
  }

  output.classList.remove('hidden');
  output.innerHTML = `<div class="text-xs text-slate-400 flex items-center space-x-2"><i data-lucide="loader-2" class="w-4 h-4 animate-spin text-emerald-400"></i><span>Tracing shortest causal path...</span></div>`;
  lucide.createIcons();

  try {
    const res = await fetch(`/api/graph/path?source_id=${encodeURIComponent(sourceId)}&target_id=${encodeURIComponent(targetId)}`);
    const data = await res.json();

    if (!data.found) {
      output.innerHTML = `<div class="text-xs text-amber-400 flex items-center space-x-2"><i data-lucide="alert-circle" class="w-4 h-4"></i><span>${data.message}</span></div>`;
    } else {
      output.innerHTML = `
        <div class="space-y-3">
          <div class="flex items-center justify-between text-xs font-semibold text-white">
            <span class="text-emerald-400 flex items-center space-x-1.5"><i data-lucide="check-circle" class="w-4 h-4"></i><span>Discovered Causal Chain (${data.hops} Hops)</span></span>
            <span class="font-mono text-slate-400 text-[11px]">${data.node_chain.join(' ➔ ')}</span>
          </div>
          <div class="space-y-2 pt-2 border-t border-slate-800">
            ${data.details.map((step, idx) => `
              <div class="p-2.5 rounded-lg bg-slate-950 text-xs flex flex-col space-y-1">
                <div class="flex items-center space-x-2">
                  <span class="w-5 h-5 rounded-full bg-emerald-900/60 text-emerald-400 font-bold flex items-center justify-center font-mono text-[10px]">${idx + 1}</span>
                  <span class="font-bold text-white">${escapeHtml(step.from.name)}</span>
                  <span class="text-emerald-400 font-mono text-[11px] font-bold">&mdash; [${step.relation}] &rarr;</span>
                  <span class="font-bold text-white">${escapeHtml(step.to.name)}</span>
                </div>
                <p class="text-slate-300 italic text-[11px] pl-7">"${escapeHtml(step.evidence)}"</p>
                <div class="text-[10px] text-slate-400 pl-7">Source: ${step.filename} (Page ${step.page})</div>
              </div>
            `).join('')}
          </div>
        </div>
      `;
    }
    lucide.createIcons();
  } catch (err) {
    console.error("Causal path error:", err);
    output.innerHTML = `<div class="text-xs text-rose-400">Error tracing path.</div>`;
  }
}

// -----------------------------------------------------------------------------
// TAB 4: MULTI-PARADIGM COMPARISON LAB
// -----------------------------------------------------------------------------
async function runComparison() {
  const query = document.getElementById('compareInput').value.trim();
  if (!query) {
    showToast("Please enter a query to compare.");
    return;
  }

  const btn = document.getElementById('compareSubmitBtn');
  btn.disabled = true;
  btn.innerHTML = `<i data-lucide="loader-2" class="w-4 h-4 animate-spin"></i><span>Executing 5 Engines...</span>`;
  lucide.createIcons();

  try {
    const res = await fetch('/api/compare', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, top_k: 3 })
    });
    const data = await res.json();
    renderComparison(data);
  } catch (err) {
    console.error("Comparison error:", err);
    showToast("Error running multi-retriever comparison.");
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<i data-lucide="play" class="w-4 h-4"></i><span>Compare All 5 Engines</span>`;
    lucide.createIcons();
  }
}

function renderComparison(data) {
  const grid = document.getElementById('compareResultsGrid');
  const details = document.getElementById('compareDetailedSection');
  grid.classList.remove('hidden');
  details.classList.remove('hidden');

  const methods = Object.keys(data.comparison);

  grid.innerHTML = methods.map(name => {
    const info = data.comparison[name];
    const topResult = info.results[0];
    const isHybrid = name === 'Hybrid GraphRAG';

    return `
      <div class="stat-card ${isHybrid ? 'border-emerald-500/50 bg-emerald-950/10' : ''} space-y-3">
        <div class="flex items-center justify-between">
          <span class="text-xs font-bold text-white uppercase">${name}</span>
          ${isHybrid ? '<span class="px-1.5 py-0.5 rounded text-[9px] bg-emerald-500 text-black font-extrabold">BEST</span>' : ''}
        </div>
        <div>
          <div class="text-2xl font-extrabold text-white">${info.latency_ms} <span class="text-xs font-normal text-slate-400">ms</span></div>
          <div class="text-[11px] text-slate-400 mt-0.5">Searched ${info.total_searched} chunks</div>
        </div>
        <div class="pt-2 border-t border-slate-800 text-xs">
          <div class="text-[10px] text-slate-400 uppercase font-semibold">Top Result Score:</div>
          <div class="font-mono text-emerald-400 font-bold text-xs mt-0.5">${topResult ? (typeof topResult.score === 'number' ? topResult.score.toFixed(4) : topResult.score) : '0.0000'}</div>
          <div class="text-[11px] text-slate-300 truncate mt-1">${topResult ? topResult.filename : 'No results'}</div>
        </div>
      </div>
    `;
  }).join('');

  // Detailed Tabs
  const tabHeaders = document.getElementById('compareTabsHeaders');
  const tabContent = document.getElementById('compareTabsContent');

  tabHeaders.innerHTML = methods.map((m, idx) => `
    <button onclick="switchCompareTab('${m}')" class="compare-subtab px-4 py-2 rounded-t-xl text-xs font-semibold ${idx === 0 ? 'bg-slate-800 text-emerald-400 border-t-2 border-emerald-500' : 'text-slate-400 hover:text-white'} transition" data-method="${m}">${m}</button>
  `).join('');

  renderCompareSubTabContent(methods[0], data.comparison[methods[0]]);
}

function switchCompareTab(methodName) {
  document.querySelectorAll('.compare-subtab').forEach(b => {
    const isSelected = b.dataset.method === methodName;
    b.className = `compare-subtab px-4 py-2 rounded-t-xl text-xs font-semibold ${isSelected ? 'bg-slate-800 text-emerald-400 border-t-2 border-emerald-500' : 'text-slate-400 hover:text-white'} transition`;
  });
  // Fetch current comparison data
  const compareInput = document.getElementById('compareInput').value.trim();
  fetch('/api/compare', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query: compareInput, top_k: 3 })
  }).then(r => r.json()).then(data => {
    renderCompareSubTabContent(methodName, data.comparison[methodName]);
  });
}

function renderCompareSubTabContent(methodName, info) {
  const container = document.getElementById('compareTabsContent');
  if (!info.results || info.results.length === 0) {
    container.innerHTML = `<div class="text-xs text-slate-400">No results found for this method.</div>`;
    return;
  }

  container.innerHTML = info.results.map((r, idx) => `
    <div class="p-3.5 rounded-xl bg-slate-900 border border-slate-800 space-y-2">
      <div class="flex items-center justify-between text-xs">
        <span class="font-bold text-emerald-400 font-mono">Rank ${idx + 1} &bull; Score: ${typeof r.score === 'number' ? r.score.toFixed(4) : r.score}</span>
        <span class="font-mono text-slate-400">${r.filename} (Page ${r.page_number})</span>
      </div>
      <p class="text-xs text-slate-300 leading-relaxed font-mono bg-slate-950 p-3 rounded-lg border border-slate-800/60">${escapeHtml(r.text)}</p>
    </div>
  `).join('');
}

// -----------------------------------------------------------------------------
// TAB 5: BENCHMARK EVALUATION HUB
// -----------------------------------------------------------------------------
async function loadBenchmarkSummary() {
  try {
    const res = await fetch('/api/benchmark/summary');
    const json = await res.json();
    if (!json.exists || !json.data || json.data.length === 0) return;

    appState.benchmarkData = json.data;
    renderBenchmarkTable(json.data);
    renderBenchmarkCharts(json.data);
  } catch (err) {
    console.error("Benchmark load error:", err);
  }
}

function renderBenchmarkTable(data) {
  const tbody = document.getElementById('benchmarkTableBody');
  if (!tbody) return;

  tbody.innerHTML = data.map(row => {
    const isHybrid = (row["Retrieval Strategy"] || row["Retrieval Method"] || "").includes("Hybrid");
    return `
      <tr class="${isHybrid ? 'bg-emerald-950/20 font-semibold' : ''} hover:bg-slate-800/40 transition">
        <td class="py-3 px-4 font-sans font-semibold text-white flex items-center space-x-2">
          ${isHybrid ? '<i data-lucide="award" class="w-4 h-4 text-emerald-400"></i>' : ''}
          <span>${row["Retrieval Strategy"] || row["Retrieval Method"]}</span>
        </td>
        <td class="py-3 px-4 text-center ${isHybrid ? 'text-emerald-400 font-bold' : ''}">${row["Precision@K"] ?? row["precision_at_k"]}</td>
        <td class="py-3 px-4 text-center ${isHybrid ? 'text-emerald-400 font-bold' : ''}">${row["Recall@K"] ?? row["recall_at_k"]}</td>
        <td class="py-3 px-4 text-center">${row["MRR"] ?? row["reciprocal_rank"]}</td>
        <td class="py-3 px-4 text-center">${row["MAP"] ?? row["average_precision"]}</td>
        <td class="py-3 px-4 text-center ${isHybrid ? 'text-emerald-400 font-bold' : ''}">${row["NDCG@K"] ?? row["ndcg_at_k"]}</td>
        <td class="py-3 px-4 text-center text-slate-400 font-mono">${row["Latency (ms)"] ?? row["latency_ms"]} ms</td>
        <td class="py-3 px-4 text-center ${isHybrid ? 'text-emerald-400 font-bold' : ''}">${row["Evidence Support Rate"] ?? row["faithfulness"]}</td>
      </tr>
    `;
  }).join('');
  lucide.createIcons();
}

function renderBenchmarkCharts(data) {
  const labels = data.map(d => d["Retrieval Strategy"] || d["Retrieval Method"]);
  const pAtK = data.map(d => d["Precision@K"] ?? d["precision_at_k"]);
  const rAtK = data.map(d => d["Recall@K"] ?? d["recall_at_k"]);
  const ndcg = data.map(d => d["NDCG@K"] ?? d["ndcg_at_k"]);

  // Bar Chart
  const ctxBar = document.getElementById('benchmarkBarChart')?.getContext('2d');
  if (ctxBar) {
    if (appState.barChart) appState.barChart.destroy();
    appState.barChart = new Chart(ctxBar, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [
          { label: 'Precision@K', data: pAtK, backgroundColor: '#3b82f6' },
          { label: 'Recall@K', data: rAtK, backgroundColor: '#10b981' },
          { label: 'NDCG@K', data: ndcg, backgroundColor: '#8b5cf6' }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { labels: { color: '#94a3b8', font: { family: 'Inter' } } } },
        scales: {
          x: { ticks: { color: '#94a3b8', font: { family: 'Inter', size: 10 } }, grid: { color: '#1e293b' } },
          y: { min: 0, max: 1.05, ticks: { color: '#94a3b8' }, grid: { color: '#1e293b' } }
        }
      }
    });
  }

  // Radar Chart
  const ctxRadar = document.getElementById('benchmarkRadarChart')?.getContext('2d');
  if (ctxRadar) {
    if (appState.radarChart) appState.radarChart.destroy();
    const colors = ['#3b82f6', '#f59e0b', '#ec4899', '#64748b', '#10b981'];
    appState.radarChart = new Chart(ctxRadar, {
      type: 'radar',
      data: {
        labels: ['Precision@K', 'Recall@K', 'MRR', 'MAP', 'NDCG@K', 'Evidence Support'],
        datasets: data.map((d, idx) => ({
          label: d["Retrieval Strategy"] || d["Retrieval Method"],
          data: [
            d["Precision@K"] ?? d["precision_at_k"],
            d["Recall@K"] ?? d["recall_at_k"],
            d["MRR"] ?? d["reciprocal_rank"],
            d["MAP"] ?? d["average_precision"],
            d["NDCG@K"] ?? d["ndcg_at_k"],
            d["Evidence Support Rate"] ?? d["faithfulness"]
          ],
          borderColor: colors[idx % colors.length],
          backgroundColor: `${colors[idx % colors.length]}20`,
          borderWidth: (d["Retrieval Strategy"] || "").includes("Hybrid") ? 3 : 1.5
        }))
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { labels: { color: '#94a3b8', font: { family: 'Inter', size: 10 } } } },
        scales: {
          r: {
            angleLines: { color: '#1e293b' },
            grid: { color: '#1e293b' },
            pointLabels: { color: '#94a3b8', font: { family: 'Inter', size: 10 } },
            ticks: { display: false }
          }
        }
      }
    });
  }
}

async function runBenchmarkEvaluation() {
  const btn = document.getElementById('runBenchmarkBtn');
  btn.disabled = true;
  btn.innerHTML = `<i data-lucide="loader-2" class="w-4 h-4 animate-spin"></i><span>Running 32 Queries (1-2 mins)...</span>`;
  lucide.createIcons();

  try {
    const res = await fetch('/api/benchmark/run', { method: 'POST' });
    const data = await res.json();
    showToast(data.message);
    loadBenchmarkSummary();
  } catch (err) {
    console.error("Evaluation run failed:", err);
    showToast("Error running benchmark.");
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<i data-lucide="play" class="w-4 h-4"></i><span>Execute 32-Query Benchmark</span>`;
    lucide.createIcons();
  }
}

async function loadBenchmarkQueries() {
  const list = document.getElementById('benchmarkQueriesList');
  if (!list) return;

  try {
    const res = await fetch('/api/benchmark/queries');
    const queries = await res.json();

    list.innerHTML = queries.map(q => `
      <div class="p-3 rounded-xl bg-slate-900 border border-slate-800 text-xs flex flex-col space-y-1">
        <div class="flex items-center justify-between">
          <span class="font-bold font-mono text-emerald-400">${q.query_id} &bull; ${q.category}</span>
          <span class="text-[11px] text-slate-400">Target Docs: ${q.target_documents.join(', ')}</span>
        </div>
        <p class="text-white font-medium">${escapeHtml(q.query)}</p>
        <div class="text-[11px] text-slate-400 flex flex-wrap gap-1 pt-1">
          <span class="text-slate-400">Expected:</span>
          ${q.expected_keywords.map(kw => `<span class="px-1.5 py-0.5 rounded bg-slate-800 text-slate-300 font-mono text-[10px]">${kw}</span>`).join('')}
        </div>
      </div>
    `).join('');
  } catch (err) {
    console.error("Benchmark queries load error:", err);
  }
}

// -----------------------------------------------------------------------------
// TAB 6: DOCUMENT REPOSITORY
// -----------------------------------------------------------------------------
async function loadDocuments() {
  const grid = document.getElementById('documentCardsGrid');
  if (!grid) return;

  try {
    const res = await fetch('/api/documents');
    const docs = await res.json();

    if (docs.length === 0) {
      grid.innerHTML = `<div class="col-span-full py-12 text-center text-slate-500 text-xs">No documents indexed. Click "Re-Ingest Authentic Default Corpus" above.</div>`;
      return;
    }

    grid.innerHTML = docs.map(d => `
      <div class="bg-surface-card border border-surface-border rounded-2xl p-5 space-y-3 flex flex-col justify-between hover:border-slate-700 transition">
        <div>
          <div class="flex items-center justify-between text-xs">
            <span class="px-2 py-0.5 rounded font-mono font-bold uppercase ${
              d.file_type === 'pdf' ? 'bg-red-950 text-red-400 border border-red-800/40' :
              d.file_type === 'docx' ? 'bg-blue-950 text-blue-400 border border-blue-800/40' :
              'bg-slate-800 text-emerald-400 border border-slate-700'
            }">${d.file_type}</span>
            <span class="text-slate-400 text-[11px] font-mono">${d.total_pages} Page${d.total_pages > 1 ? 's' : ''} &bull; ${d.chunk_count} Chunks</span>
          </div>
          <h4 class="font-bold text-white text-sm mt-2 leading-snug">${escapeHtml(d.title)}</h4>
          <div class="text-xs text-slate-400 font-mono truncate mt-1">${escapeHtml(d.filename)}</div>
          <div class="text-[10px] text-slate-400 font-mono mt-2">SHA: ${d.file_hash.substring(0, 16)}...</div>
        </div>
        <div class="pt-3 border-t border-slate-800 flex items-center justify-between text-xs">
          <button onclick="viewDocChunks('${d.doc_id}')" class="text-emerald-400 hover:text-emerald-300 font-medium flex items-center space-x-1">
            <i data-lucide="eye" class="w-3.5 h-3.5"></i>
            <span>View Chunks</span>
          </button>
          <button onclick="deleteDoc('${d.doc_id}')" class="text-rose-400 hover:text-rose-300 p-1.5 rounded-lg hover:bg-rose-950/40 transition">
            <i data-lucide="trash-2" class="w-3.5 h-3.5"></i>
          </button>
        </div>
      </div>
    `).join('');
    lucide.createIcons();
  } catch (err) {
    console.error("Failed to load documents:", err);
  }
}

async function viewDocChunks(docId) {
  try {
    const res = await fetch(`/api/documents/${docId}`);
    const data = await res.json();

    document.getElementById('modalDocTitle').innerText = data.document.title;
    document.getElementById('modalDocFilename').innerText = `${data.document.filename} (${data.chunks.length} chunks)`;
    
    const body = document.getElementById('modalChunksBody');
    body.innerHTML = data.chunks.map(c => `
      <div class="p-3.5 rounded-xl bg-slate-900 border border-slate-800 space-y-1.5">
        <div class="flex items-center justify-between text-[11px] text-slate-400 font-mono">
          <span class="text-emerald-400 font-bold">${c.chunk_id} (Page ${c.page_number})</span>
          <span>${c.word_count} words &bull; chars [${c.char_start}..${c.char_end}]</span>
        </div>
        <p class="text-slate-200 text-xs font-mono leading-relaxed bg-slate-950 p-2.5 rounded border border-slate-800/80">${escapeHtml(c.text)}</p>
      </div>
    `).join('');

    document.getElementById('docChunksModal').classList.remove('hidden');
  } catch (err) {
    console.error("View chunks error:", err);
    showToast("Failed to load document chunks.");
  }
}

function closeDocModal() {
  document.getElementById('docChunksModal').classList.add('hidden');
}

async function deleteDoc(docId) {
  if (!confirm("Are you sure you want to delete this document? All associated chunks and Knowledge Graph relations will be pruned.")) return;

  try {
    const res = await fetch(`/api/documents/${docId}`, { method: 'DELETE' });
    const data = await res.json();
    showToast(data.message);
    loadDocuments();
    loadStats();
  } catch (err) {
    console.error("Delete doc error:", err);
    showToast("Error deleting document.");
  }
}

async function uploadFiles() {
  const fileInput = document.getElementById('docFileInput');
  if (!fileInput.files || fileInput.files.length === 0) {
    showToast("Please choose at least one PDF, DOCX, or TXT file.");
    return;
  }

  const formData = new FormData();
  for (let i = 0; i < fileInput.files.length; i++) {
    formData.append('files', fileInput.files[i]);
  }
  formData.append('chunk_size', document.getElementById('uploadChunkSize').value);
  formData.append('chunk_overlap', document.getElementById('uploadChunkOverlap').value);

  const btn = document.getElementById('uploadBtn');
  btn.disabled = true;
  btn.innerHTML = `<i data-lucide="loader-2" class="w-4 h-4 animate-spin"></i><span>Extracting Chunks & KG...</span>`;
  lucide.createIcons();

  try {
    const res = await fetch('/api/documents/upload', {
      method: 'POST',
      body: formData
    });
    const data = await res.json();
    showToast(`Successfully ingested ${data.ingested_count} files (${data.chunks_created} chunks)!`);
    fileInput.value = '';
    loadDocuments();
    loadStats();
  } catch (err) {
    console.error("Upload error:", err);
    showToast("Failed to upload and process documents.");
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<i data-lucide="plus-circle" class="w-4 h-4"></i><span>Process & Extract Knowledge Graph</span>`;
    lucide.createIcons();
  }
}

async function ingestDefaultCorpus() {
  showToast("Re-ingesting authentic climate corpus and rebuilding Knowledge Graph...");
  try {
    const res = await fetch('/api/documents/ingest-default', { method: 'POST' });
    const data = await res.json();
    showToast(`Corpus ready: ${data.total_documents} documents, ${data.total_chunks} chunks.`);
    loadDocuments();
    loadStats();
  } catch (err) {
    console.error("Default ingest error:", err);
    showToast("Error re-ingesting corpus.");
  }
}

// -----------------------------------------------------------------------------
// TAB 7: SYSTEM DIAGNOSTICS & SCHEMA
// -----------------------------------------------------------------------------
async function loadDiagnostics() {
  try {
    const res = await fetch('/api/stats');
    const data = await res.json();

    const neoBadge = document.getElementById('diagNeo4jBadge');
    if (neoBadge) {
      if (data.neo4j_connected) {
        neoBadge.className = 'px-2.5 py-1 rounded-full bg-emerald-900/60 text-emerald-400 border border-emerald-700/50 font-semibold text-[10px]';
        neoBadge.innerText = 'Connected (bolt://localhost:7687)';
      } else {
        neoBadge.className = 'px-2.5 py-1 rounded-full bg-slate-800 text-slate-400 border border-slate-700 font-semibold text-[10px]';
        neoBadge.innerText = 'Offline (Fallback Active)';
      }
    }

    const ollamaBadge = document.getElementById('diagOllamaBadge');
    if (ollamaBadge) {
      if (data.ollama_available) {
        ollamaBadge.className = 'px-2.5 py-1 rounded-full bg-emerald-900/60 text-emerald-400 border border-emerald-700/50 font-semibold text-[10px]';
        ollamaBadge.innerText = 'Online (llama3)';
      } else {
        ollamaBadge.className = 'px-2.5 py-1 rounded-full bg-slate-800 text-slate-400 border border-slate-700 font-semibold text-[10px]';
        ollamaBadge.innerText = 'Offline (Deterministic Extractive Synthesis Active)';
      }
    }

    // Schema Node Types List
    const schemaNodes = [
      { name: 'ClimatePhenomenon', desc: 'Meteorological and physical dynamics (Global Warming, Sea Level Rise, Heatwaves, Ocean Acidification).' },
      { name: 'GreenhouseGas', desc: 'Radiative forcing atmospheric gases (Carbon Dioxide, Methane, Nitrous Oxide).' },
      { name: 'Region', desc: 'Geographic entities and biomes (India, South Asia, Arctic, Amazon Basin, Global).' },
      { name: 'ClimateImpact', desc: 'Downstream consequences (Agricultural Yield Decline, Water Scarcity, Coral Bleaching).' },
      { name: 'MitigationStrategy', desc: 'Decarbonization measures (Solar Energy, Wind Energy, Afforestation, CCS).' },
      { name: 'AdaptationStrategy', desc: 'Resilience mechanisms (Drip Irrigation, Drought-Resilient Crop Varieties, Rainwater Harvesting).' },
      { name: 'Policy', desc: 'Treaties, plans and legal targets (Paris Agreement, Kyoto Protocol, NAPCC, NDCs).' },
      { name: 'ResearchPaper', desc: 'Authoritative publications (IPCC Sixth Assessment Report, UNEP Emissions Gap Report).' },
      { name: 'Organization', desc: 'International bodies and research institutes (IPCC, UNFCCC, WMO, UNEP, ICAR).' },
      { name: 'Dataset', desc: 'Climate observations and models (ERA5 Reanalysis, CMIP6 Projections, NOAA Mauna Loa CO2).' }
    ];

    const schemaRelations = [
      { name: 'CAUSES', desc: 'Direct physical causation: Greenhouse Gases CAUSES Global Warming.' },
      { name: 'AFFECTS', desc: 'Ecological and societal impact: Global Warming AFFECTS Agricultural Yield Decline.' },
      { name: 'OCCURS_IN', desc: 'Spatial grounding: Agricultural Yield Decline OCCURS_IN India.' },
      { name: 'MITIGATED_BY', desc: 'Emission reduction pathway: Global Warming MITIGATED_BY Solar Energy.' },
      { name: 'ADAPTED_BY', desc: 'Resilience intervention: Agricultural Yield Decline ADAPTED_BY Drip Irrigation.' },
      { name: 'RECOMMENDED_BY', desc: 'Institutional authority: Drought-Resilient Varieties RECOMMENDED_BY ICAR.' },
      { name: 'SUPPORTED_BY', desc: 'Empirical backing: Paris Agreement SUPPORTED_BY IPCC AR6.' },
      { name: 'RELATED_TO', desc: 'General verified contextual link with sentence evidence excerpt.' }
    ];

    document.getElementById('schemaNodeTypesList').innerHTML = schemaNodes.map(n => `
      <div class="p-2.5 rounded-lg bg-slate-900 border border-slate-800 space-y-0.5">
        <span class="font-bold text-emerald-400 font-mono">${n.name}</span>
        <p class="text-slate-300 text-[11px]">${n.desc}</p>
      </div>
    `).join('');

    document.getElementById('schemaRelationTypesList').innerHTML = schemaRelations.map(r => `
      <div class="p-2.5 rounded-lg bg-slate-900 border border-slate-800 space-y-0.5">
        <span class="font-bold text-indigo-400 font-mono">${r.name}</span>
        <p class="text-slate-300 text-[11px]">${r.desc}</p>
      </div>
    `).join('');
  } catch (err) {
    console.error("Diagnostics load error:", err);
  }
}

// -----------------------------------------------------------------------------
// Utilities
// -----------------------------------------------------------------------------
function showToast(msg) {
  const toast = document.getElementById('toastNotification');
  document.getElementById('toastMsg').innerText = msg;
  toast.classList.remove('opacity-0', 'translate-y-20', 'pointer-events-none');
  setTimeout(() => {
    toast.classList.add('opacity-0', 'translate-y-20', 'pointer-events-none');
  }, 3500);
}

function escapeHtml(text) {
  if (!text) return '';
  return String(text)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

function debounce(func, wait) {
  let timeout;
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
}
