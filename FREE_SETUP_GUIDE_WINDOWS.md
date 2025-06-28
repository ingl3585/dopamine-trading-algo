# 🆓 Free AI Trading Personality Setup Guide - Windows

Complete step-by-step guide to get the AI Trading Personality system running **completely free** on Windows.

## 🎯 What You'll Get for Free

✅ **Complete AI personality system** with human-like emotions  
✅ **Local LLM** (Llama2 - no API costs)  
✅ **Windows voice synthesis** (built-in SAPI)  
✅ **Memory and learning** capabilities  
✅ **Full trading integration** ready  
✅ **No monthly fees** or API keys required  
✅ **Offline capable** once set up  

## 📋 Prerequisites

- Windows 10/11
- Python 3.8+ installed
- PowerShell or Command Prompt access
- 8GB+ RAM (for Ollama)
- 5GB+ free disk space

## 🚀 Step-by-Step Installation

### Step 1: Install Ollama (Free Local LLM)

**Option A: Official Windows Installer (Recommended)**
1. Go to https://ollama.ai/download
2. Click "Download for Windows"
3. Run the downloaded `OllamaSetup.exe`
4. Follow the installer prompts

**Option B: Manual Installation**
1. Download the Windows executable from GitHub releases
2. Place it in a folder like `C:\ollama`
3. Add to Windows PATH environment variable

**Verify Installation:**
```powershell
ollama --version
```

### Step 2: Download Free AI Model

```powershell
# Download the free Llama2 7B model (one-time ~4GB download)
ollama pull llama2:7b-chat

# Test the model
ollama run llama2:7b-chat
# Type: "Hello, how are you today?"
# Type: "/bye" to exit
```

**Alternative Smaller Models (if you have limited RAM):**
```powershell
# Smaller models for lower-end systems
ollama pull llama2:3b-instruct  # ~2GB
ollama pull phi:latest          # ~1.6GB
```

### Step 3: Install Python Dependencies

```powershell
# Navigate to your project directory
cd "C:\Users\ingle\OneDrive\Desktop\Actor_Critic_ML_NT"

# Install required packages for free setup
pip install numpy scipy pandas rich aiohttp pyttsx3

# Verify installations
python -c "import numpy, scipy, pandas, rich, aiohttp, pyttsx3; print('All packages installed successfully!')"
```

### Step 4: Configure Free Setup

**Copy the free configuration:**
```powershell
# Copy the free configuration to main config file
copy config\personality_config_free.json config\personality_config.json
```

**Or create manually:**
Create `config/personality_config.json`:
```json
{
  "personality": {
    "enabled": true,
    "personality_name": "Alex",
    "auto_commentary": true,
    "commentary_interval": 60.0,
    "log_commentary": true
  },
  "llm": {
    "model_name": "llama2:7b-chat",
    "api_key": "",
    "base_url": "http://localhost:11434",
    "temperature": 0.7,
    "max_tokens": 200
  },
  "voice": {
    "enabled": true,
    "service": "local",
    "speed": 1.0,
    "volume": 0.8
  },
  "development": {
    "mock_llm": false,
    "mock_voice": false,
    "debug_logging": true
  }
}
```

### Step 5: Start Ollama Server

**Open a new PowerShell window and keep it running:**
```powershell
ollama serve
```

You should see:
```
2024/01/01 12:00:00 routes.go:777: Listening on 127.0.0.1:11434 (version 0.1.17)
```

### Step 6: Test the Free Setup

**Run the comprehensive test:**
```powershell
python test_free_personality.py
```

**Expected output:**
```
🆓 FREE AI Trading Personality Test Suite
==================================================

📦 Testing Module Imports
-------------------------
✅ trading_personality imported
✅ config_manager imported
...

🎉 ALL TESTS PASSED! Your free AI Trading Personality setup is ready!
```

### Step 7: Run the Demo

```powershell
python personality_demo.py
```

**You should see:**
```
🤖 Initializing AI Trading Personality Demo
==================================================
Configuration Status: valid
Personality: Alex
LLM Model: llama2:7b-chat
Voice Enabled: true
...
```

## 🔧 Troubleshooting

### ❌ "sh is not recognized" Error
You got this error because you tried the Linux command on Windows. Use the Windows installer instead from https://ollama.ai/download

### ❌ Ollama Won't Start
```powershell
# Check if port is in use
netstat -an | findstr 11434

# Kill any existing Ollama processes
taskkill /f /im ollama.exe

# Restart Ollama
ollama serve
```

### ❌ Model Download Fails
```powershell
# Try a different model
ollama pull phi:latest

# Or use smaller model
ollama pull llama2:3b-instruct

# Update your config file to match the model name
```

### ❌ Python Import Errors
```powershell
# Reinstall packages
pip uninstall numpy scipy pandas rich aiohttp pyttsx3
pip install numpy scipy pandas rich aiohttp pyttsx3

# Check Python version
python --version  # Should be 3.8+
```

### ❌ Voice Synthesis Not Working
```powershell
# Test pyttsx3 separately
python -c "import pyttsx3; engine = pyttsx3.init(); engine.say('Hello'); engine.runAndWait()"

# If it fails, install SAPI voices
# Go to Settings > Time & Language > Speech > Manage voices
```

### ❌ LLM Connection Fails
1. **Check Ollama server is running**
2. **Verify the model is downloaded:** `ollama list`
3. **Test manually:** `ollama run llama2:7b-chat`
4. **Use mock mode as fallback:**
   ```json
   {
     "development": {
       "mock_llm": true
     }
   }
   ```

## 🎛️ Fallback: Pure Mock Mode

If Ollama has issues, use pure mock mode:

```powershell
# Copy mock configuration
copy config\personality_config_mock.json config\personality_config.json

# Test mock mode
python test_free_personality.py
```

Mock mode gives you:
- ✅ All personality features
- ✅ Emotional states and memory
- ✅ Pre-written responses
- ✅ Voice synthesis
- ✅ Complete integration framework

## 📊 Performance Optimization

### For Lower-End Systems:
```json
{
  "llm": {
    "model_name": "phi:latest",
    "max_tokens": 100
  },
  "memory": {
    "short_term_maxlen": 25,
    "session_maxlen": 15
  }
}
```

### For High-End Systems:
```json
{
  "llm": {
    "model_name": "llama2:13b-chat",
    "max_tokens": 300
  },
  "personality": {
    "commentary_interval": 30.0
  }
}
```

## 🔄 Daily Usage

### Start Your Trading Day:
```powershell
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Run your trading system with personality
python main.py  # Your main trading script
```

### Monitor Personality:
```powershell
# Check personality status
python -c "
import sys; sys.path.insert(0, 'src')
from personality.personality_integration import PersonalityIntegration
integration = PersonalityIntegration()
print(integration.get_personality_status())
"
```

## 🎯 Next Steps

1. **✅ Complete this free setup**
2. **📖 Read** `PERSONALITY_INTEGRATION_GUIDE.md`
3. **🔗 Integrate** with your trading system
4. **🎨 Customize** personality traits and responses
5. **📈 Monitor** performance and adapt settings

## 💰 Cost Comparison

| Feature | Free Setup | Premium APIs |
|---------|------------|--------------|
| **LLM** | Ollama (Free) | OpenAI ($20+/month) |
| **Voice** | Windows SAPI (Free) | ElevenLabs ($5+/month) |
| **Memory** | Local files (Free) | Cloud storage ($) |
| **Total** | **$0/month** | **$25+/month** |

## 🎉 Success Indicators

You'll know it's working when you see:
- ✅ Ollama server running on port 11434
- ✅ Test script passes all checks
- ✅ Personality responds to queries
- ✅ Voice synthesis speaks responses
- ✅ Memory system tracks interactions
- ✅ Emotional states change based on context

**Congratulations! You now have a completely free AI Trading Personality system running locally on your Windows machine!**

## 🤝 Getting Help

If you encounter issues:
1. Check this troubleshooting section
2. Run `python test_free_personality.py` for diagnostics
3. Check Ollama logs: look in the terminal where you ran `ollama serve`
4. Try mock mode as a fallback

The system is designed to gracefully degrade - if one component fails, others continue working.