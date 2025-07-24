# 🎯 ImageWeaver

**AI-Powered Image Placement for Translated Documents**

ImageWeaver intelligently restores images to translated HTML files using advanced context analysis and LLM refinement. Perfect for Korean→English translation workflows where images get lost during text-only translation.

## 🚀 Quick Start (Windows)

### One-Click Setup & Launch
```cmd
# Download/clone repository, then simply run:
start.bat
```
This will automatically:
1. Set up Python virtual environment
2. Install all dependencies
3. Launch ImageWeaver GUI

### Manual Setup
```cmd
setup.bat          # One-time setup
run_gui.bat         # Launch GUI
run_console.bat     # Launch console version
```

## ✨ Features

- **🎯 Advanced Context Extraction**: Dynamic thresholds, quality scoring, adaptive windowing
- **🧠 LLM-Powered Refinement**: Uses Ollama (local) or OpenAI for intelligent placement decisions  
- **📊 Smart Fallback Strategies**: 4-tier placement system ensures images are never lost
- **🖥️ Modern Dark Mode GUI**: Professional interface with real-time progress tracking
- **💻 Command Line Support**: Batch processing for automation workflows
- **📈 Detailed Analytics**: Success rates, strategy breakdowns, performance metrics
- **🚀 Standalone Executable**: No Python installation required

## 🎯 How It Works

1. **Scan**: Original HTML files (with images) + Translated HTML files (text only)
2. **Extract**: Advanced context analysis around each image position
3. **Match**: LLM analyzes Korean context to find best English paragraph match
4. **Place**: Intelligent placement with confidence scoring and fallback strategies
5. **Report**: Detailed success metrics and strategy analysis

## 🛠️ Installation Options

### Option 1: Download Executable (Recommended)
1. Download the latest release from [Releases](../../releases)
2. Extract and run `ImageWeaver.exe` (Windows) or `ImageWeaver` (macOS/Linux)
3. No Python installation required!

### Option 2: Run from Source (Windows)
```cmd
# Clone repository
git clone https://github.com/yourusername/ImageWeaver.git
cd ImageWeaver

# One-click setup and launch
start.bat
```

### Option 3: Run from Source (Cross-platform)
```bash
# Clone repository
git clone https://github.com/yourusername/ImageWeaver.git
cd ImageWeaver

# Install dependencies
pip install -r requirements.txt

# Run GUI
python imageweaver_gui.py

# Or run console version
python imageweaver_console.py --help
```

## 🎯 LLM Configuration

### Ollama (Recommended - Free & Local)
1. Install [Ollama](https://ollama.ai)
2. Pull model: `ollama pull llama3.2:1b`
3. Start server: `ollama serve`
4. Test connection in ImageWeaver

### OpenAI (Cloud - Paid)
1. Get API key from [OpenAI](https://platform.openai.com)
2. Enter key in LLM Config tab
3. Uses gpt-4o-mini for cost efficiency

## 🛠️ Development Commands (Windows)

```cmd
setup.bat           # Initial setup
run_gui.bat         # Launch GUI
run_console.bat     # Launch console
activate_env.bat    # Development environment
build.bat           # Build executable
test.bat            # Run tests
start.bat           # Quick start menu
```

## 📊 Performance

- **15-25x faster** than brute-force approaches
- **Dynamic thresholds** adapt to document characteristics
- **Quality scoring** ensures rich context for LLM decisions
- **95% cost reduction** with focused targeting approach

## 🔧 Advanced Features

- **Dynamic Threshold Adaptation**: 60-100 chars based on paragraph length
- **Context Quality Scoring**: Content diversity + dialogue/action detection  
- **Adaptive Window Sizing**: 3-8 paragraphs based on document structure
- **Intelligent Context Usage**: Up to 5+5 paragraphs for LLM analysis
- **Repetitive Content Detection**: Automatic fallback strategies

## 📁 File Matching Patterns

ImageWeaver automatically matches files using these patterns:
- `chapter01.html` → `chapter01.html`
- `chapter01.html` → `chapter01_translated.html`
- `chapter01.html` → `chapter01_ko_to_en_20241216.html`
- `chapter01.html` → `chapter01_en.html`

## 🎯 Supported Formats

- **Input**: HTML, HTM files
- **Images**: JPG, PNG, GIF, SVG (embedded in HTML)
- **Output**: HTML with properly placed images

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- [Issues](../../issues) - Report bugs or request features
- [Releases](../../releases) - Download latest version
- [Wiki](../../wiki) - Detailed documentation

## 💡 Use Cases

- **Post-Translation Processing**: Restore images after AI translation
- **Content Migration**: Move images between different HTML structures  
- **Quality Assurance**: Ensure translated documents maintain visual layout
- **Batch Operations**: Process large sets of translated files efficiently

---

**Made with ❤️ for the translation community**