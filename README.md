# Marswalk AI Assistant

> **JARVIS-style intelligent mission control system for Mars exploration**

A local-first AI assistant that integrates with the Marswalk Mission Planner to provide natural language interaction, route planning, terrain analysis, scientific knowledge retrieval, and map control — all running on your laptop with zero cloud dependencies.

![Marswalk AI](https://img.shields.io/badge/Marswalk-AI%20Assistant-red?style=for-the-badge&logo=mars)
![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green?style=flat&logo=fastapi)
![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-orange?style=flat&logo=ollama)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat)

## 🚀 Features

### Core Capabilities
- **Natural Language Interface** — Chat with the assistant using plain English
- **Tool Calling** — 18+ built-in tools for routing, science, map control, and knowledge
- **Local LLM** — Runs on Ollama with Qwen3 (1.7B/4B) or Llama 3.2 — no API keys needed
- **Conversation Memory** — SQLite-backed persistent chat history
- **Mission Context** — Tracks routes, locations, objectives, EVA status across sessions

### Scientific Tools
- `generate_route` — Create hazard-avoiding exploration routes with slope constraints
- `compare_routes` — Side-by-side route analysis with recommendations
- `analyse_terrain` — Elevation, slope, and data source at any coordinate
- `calculate_slope` — Precise slope computation between points
- `analyse_environment` — Radiation dose, temperature, pressure, wind estimates
- `retrieve_mission_information` — Perseverance, Jezero, Mars missions, geology, MEDA, radiation
- `search_knowledge_base` — RAG search over ingested scientific documents

### Map Integration
- `highlight_map_location` — Mark points of interest on the Marswalk Leaflet/3D map
- `zoom_map` — Navigate to coordinates or fit bounds
- `display_route_on_map` — Render slope-colored routes with elevation profiles
- `change_map_layer` — Toggle NASA imagery layers (Viking, CTX, HiRISE, MOLA, THEMIS)
- `navigate_application` — Switch tabs (layers, route, environment, records) and views (2D/3D)

### Knowledge System
- **Document Ingestion** — PDF, TXT, Markdown, JSON support
- **Vector Search** — ChromaDB with sentence-transformers embeddings
- **Auto-chunking** — Smart text segmentation with overlap
- **Persistent Storage** — SQLite metadata + ChromaDB vectors

## 📋 Requirements

| Component | Version | Notes |
|-----------|---------|-------|
| Python | 3.11+ | Required for FastAPI and dependencies |
| Ollama | 0.4+ | Local LLM runtime |
| RAM | 8 GB+ | 4 GB for model + 4 GB for system |
| Disk | 5 GB+ | Models + ChromaDB + documents |
| OS | Windows 10/11, Linux, macOS | Tested on Windows 11 |

### Recommended Models (CPU Optimized)
| Model | Size | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| `qwen3:1.7b` | 1.4 GB | ⚡⚡⚡ Fast | Good | Default, quick responses |
| `qwen3:4b` | 2.6 GB | ⚡⚡ Balanced | Better | Complex reasoning |
| `llama3.2:latest` | 2.0 GB | ⚡⚡ Fast | Good | General purpose |

## 🛠 Installation

### 1. Install Ollama
```bash
# Windows (PowerShell)
winget install Ollama.Ollama

# Or download from https://ollama.com/download
```

### 2. Pull Required Models
```bash
# Default model (fast, 1.4 GB)
ollama pull qwen3:1.7b

# Optional: larger model for better reasoning
ollama pull qwen3:4b

# Optional: alternative model
ollama pull llama3.2:latest
```

### 3. Start Ollama Service
```bash
# Windows - runs automatically after install
# Linux/macOS
ollama serve
```

### 4. Set Up Marswalk AI
```bash
# Navigate to project
cd C:\Ai nasa\marswalk-ai

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r backend/requirements.txt

# Copy environment config
copy backend\.env.example backend\.env
# Edit backend\.env if needed
```

### 5. Run the Backend
```bash
cd backend
python main.py
# Server starts at http://localhost:8000
```

### 6. Open the Frontend
Open `frontend/index.html` in your browser, or serve it:
```bash
cd frontend
python -m http.server 3000
# Then open http://localhost:3000
```

## 🎯 Quick Start

1. **Start Ollama**: `ollama serve` (if not auto-started)
2. **Start Backend**: `cd backend && python main.py`
3. **Open Frontend**: Open `frontend/index.html` in browser
4. **Chat**: Try these commands:
   - *"Generate a route from the habitat to stop 2"*
   - *"Analyze the terrain at the delta front"*
   - *"Compare Route A and Route B"*
   - *"What's the radiation risk for an 8-hour EVA?"*
   - *"Show me the Perseverance mission summary"*

## 🏗 Architecture

```
marswalk-ai/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── config.py            # Pydantic settings
│   ├── models.py            # SQLAlchemy models
│   ├── schemas.py           # Pydantic request/response models
│   ├── llm.py               # Ollama client wrapper
│   ├── tools.py             # Tool registry + 18 tools
│   ├── rag.py               # ChromaDB knowledge base
│   ├── requirements.txt     # Python dependencies
│   └── .env.example         # Environment template
├── frontend/
│   ├── index.html           # JARVIS-style UI
│   ├── styles.css           # Dark theme with Mars accents
│   └── app.js               # Frontend application logic
├── data/                    # Runtime data (gitignored)
│   ├── chroma/              # Vector database
│   ├── documents/           # Ingested documents
│   └── marswalk_ai.db       # SQLite database
├── docs/                    # Documentation
└── tests/                   # Test suite
```

### Data Flow
```
User Input → FastAPI /chat → Build Context → Ollama LLM
                                    ↓
                            Tool Calls? → Execute Tools → Frontend Actions
                                    ↓
                            Save to SQLite → Return Response
```

## 🔧 Configuration

Edit `backend/.env` to customize:

```env
# Model
DEFAULT_MODEL=qwen3:1.7b
OLLAMA_HOST=http://localhost:11434

# Embeddings
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Database
DATABASE_URL=sqlite:///./data/marswalk_ai.db

# Knowledge Base
CHROMA_PERSIST_DIR=./data/chroma
DOCUMENTS_DIR=./data/documents
CHUNK_SIZE=500
CHUNK_OVERLAP=50

# Conversation
MAX_HISTORY_MESSAGES=20
MAX_CONTEXT_TOKENS=4000
```

## 📚 API Reference

### Chat
```
POST /chat
{
  "message": "Generate a route to the delta",
  "conversation_id": 1,
  "use_tools: true,
  "temperature": 0.7
}

Response:
{
  "message": "I've generated Route A...",
  "conversation_id": 1,
  "tool_calls": [...],
  "tool_results": [...],
  "actions": [{"action": "display_route", ...}]
}
```

### Conversations
```
GET  /conversations              # List all
POST /conversations              # Create new
GET  /conversations/{id}         # Get details
GET  /conversations/{id}/messages
DELETE /conversations/{id}       # Archive
```

### Mission Context
```
GET  /conversations/{id}/context
PATCH /conversations/{id}/context
```

### Knowledge Base
```
POST /knowledge/ingest           # Ingest single file
POST /knowledge/ingest-directory # Ingest folder
GET  /knowledge/documents        # List documents
DELETE /knowledge/documents/{id}
POST /knowledge/search           # Vector search
```

### Tools
```
GET  /tools                      # List all tools
POST /tools/execute              # Execute tool directly
```

### Models
```
GET  /models                     # List Ollama models
POST /models/switch              # Switch model
POST /models/pull                # Pull new model
```

### Health
```
GET /health                      # System status
```

## 🧪 Testing

```bash
cd backend
python -m pytest tests/ -v

# Run specific test
python -m pytest tests/test_tools.py -v
python -m pytest tests/test_rag.py -v
```

### Test Coverage
- Tool registry and execution
- LLM integration (mocked)
- RAG ingestion and search
- Database operations
- API endpoints

## 🔌 Marswalk Integration

The assistant is designed to embed in the Marswalk Mission Planner. When running inside Marswalk, it detects the global APIs:

```javascript
// Available in Marswalk context
window.map              // Leaflet map instance
window.Mars3D           // 3D globe API
window.selectPoint2D    // Select location on 2D map
window.drawRoute        // Display route
window.setState         // Update app state
window.setView          // Switch 2D/3D
window.nasaLayers       // NASA imagery layers
window.overlays         // Backend overlays
```

### Standalone Mode
When not in Marswalk, the assistant shows notifications for map actions and works in chat-only mode.

### Embedding in Marswalk
```html
<!-- In Marswalk index.html -->
<iframe src="http://localhost:3000" id="aiAssistant" style="width: 400px; height: 100%; border: none;"></iframe>

<!-- Or mount as panel -->
<div id="aiPanel"></div>
<script type="module" src="http://localhost:3000/app.js"></script>
```

## 📖 Usage Examples

### Route Planning
```
User: "Plan a route from the habitat to explore the western delta front, max 15° slope"
AI: [Calls generate_route with waypoints] → Displays Route A on map with elevation profile
```

### Terrain Analysis
```
User: "What's the slope at 18.4589, 77.4062?"
AI: [Calls analyse_terrain] → Returns elevation -2520.9m, slope 4.2°, source: HiRISE DTM
```

### Route Comparison
```
User: "Compare the two routes"
AI: [Calls compare_routes] → Returns analysis: Route A shorter, Route B lower max slope
```

### Scientific Questions
```
User: "Tell me about the Jezero delta geology"
AI: [Calls retrieve_mission_information] → Returns delta stratigraphy, minerals, scientific importance
```

### Map Control
```
User: "Zoom into the landing site"
AI: [Calls zoom_map with landing coordinates] → Map centers on Octavia E. Butler Landing
```

## 🐛 Troubleshooting

### Ollama Connection Failed
```bash
# Check Ollama is running
ollama list

# Restart service
ollama serve
```

### Model Not Found
```bash
# Pull the model
ollama pull qwen3:1.7b

# Or switch in settings UI
```

### ChromaDB Errors
```bash
# Clear and reinitialize
rm -rf data/chroma
# Restart backend
```

### Port Conflicts
```env
# Change in .env
PORT=8001
# Update frontend API_BASE in app.js
```

### Memory Issues
- Use `qwen3:1.7b` (smallest)
- Reduce `MAX_CONTEXT_TOKENS`
- Close other applications

## 🗺 Roadmap

### Phase 1 ✅ - Local AI Foundation
- [x] Ollama integration
- [x] FastAPI backend
- [x] SQLite conversation memory
- [x] JARVIS frontend

### Phase 2 ✅ - Intelligent Assistant
- [x] Tool calling system (18 tools)
- [x] Mission context tracking
- [x] Structured responses with frontend actions

### Phase 3 ✅ - Knowledge System
- [x] ChromaDB RAG
- [x] Document ingestion pipeline
- [x] Scientific knowledge base

### Phase 4 ✅ - Application Actions
- [x] Marswalk integration adapter
- [x] Mock tools for standalone mode
- [x] Leaflet/3D map action interface

### Phase 5 🔄 - Full Integration (In Progress)
- [ ] Connect real Marswalk `/api/*` endpoints
- [ ] Real terrain data from backend
- [ ] Actual route generation via `/api/route/auto`
- [ ] Live MEDA/space weather data

### Phase 6 📋 - Optimization
- [ ] Response caching
- [ ] Tool selection improvement
- [ ] Multi-model routing (fast/slow)
- [ ] Benchmark suite

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open Pull Request

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- **NASA Mars 2020 Mission** — Perseverance rover data and imagery
- **Marswalk Team** — Original mission planner application
- **Ollama** — Local LLM runtime
- **ChromaDB** — Vector database
- **FastAPI** — Modern Python web framework

## 📞 Support

- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Documentation**: `/docs` folder

---

**Marswalk AI Assistant** — *Your intelligent companion for Mars exploration* 🚀🔴