#!/usr/bin/env python3
"""
Quick Setup Script for Free AI Trading Personality on Windows

Automates the setup process for the free configuration
"""

import os
import sys
import subprocess
import json
import shutil
from pathlib import Path

def print_header(title):
    """Print formatted header"""
    print(f"\n🔧 {title}")
    print("=" * (len(title) + 3))

def print_step(step, description):
    """Print formatted step"""
    print(f"\n{step}. {description}")

def check_python():
    """Check Python version"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} is too old (need 3.8+)")
        return False

def install_dependencies():
    """Install required Python packages"""
    packages = ['numpy', 'scipy', 'pandas', 'rich', 'aiohttp', 'pyttsx3']
    
    print("Installing Python packages...")
    for package in packages:
        try:
            print(f"  Installing {package}...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', package], 
                          check=True, capture_output=True)
            print(f"  ✅ {package} installed")
        except subprocess.CalledProcessError as e:
            print(f"  ❌ Failed to install {package}: {e}")
            return False
    
    return True

def setup_directories():
    """Create necessary directories"""
    directories = ['data', 'data/audio', 'config']
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Directory {directory} ready")

def setup_configuration():
    """Set up configuration files"""
    
    # Check if main config exists
    main_config = Path('config/personality_config.json')
    free_config = Path('config/personality_config_free.json')
    
    if not main_config.exists() and free_config.exists():
        print("Copying free configuration to main config...")
        shutil.copy(free_config, main_config)
        print("✅ Configuration copied")
    elif main_config.exists():
        print("✅ Configuration already exists")
    else:
        print("❌ No configuration files found")
        return False
    
    # Validate configuration
    try:
        with open(main_config, 'r') as f:
            config_data = json.load(f)
        print("✅ Configuration is valid JSON")
        
        # Show key settings
        personality_name = config_data.get('personality', {}).get('personality_name', 'Unknown')
        llm_model = config_data.get('llm', {}).get('model_name', 'Unknown')
        voice_enabled = config_data.get('voice', {}).get('enabled', False)
        
        print(f"  Personality: {personality_name}")
        print(f"  LLM Model: {llm_model}")
        print(f"  Voice: {'Enabled' if voice_enabled else 'Disabled'}")
        
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ Configuration JSON error: {e}")
        return False

def check_ollama():
    """Check Ollama installation and status"""
    
    # Check if ollama command exists
    try:
        result = subprocess.run(['ollama', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✅ Ollama installed: {result.stdout.strip()}")
        else:
            print("❌ Ollama command failed")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ Ollama not found")
        print("📥 Download from: https://ollama.ai/download")
        return False
    
    # Check if service is running
    try:
        result = subprocess.run(['curl', '-s', 'http://localhost:11434/api/version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ Ollama service is running")
        else:
            print("⚠️ Ollama service not running")
            print("💡 Start with: ollama serve")
    except:
        print("⚠️ Cannot check Ollama service status")
    
    # Check models
    try:
        result = subprocess.run(['ollama', 'list'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and 'llama2' in result.stdout.lower():
            print("✅ Llama2 model available")
        else:
            print("⚠️ No Llama2 model found")
            print("💡 Install with: ollama pull llama2:7b-chat")
    except:
        print("⚠️ Cannot check Ollama models")
    
    return True

def test_voice():
    """Test voice synthesis"""
    try:
        import pyttsx3
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        
        if voices:
            print(f"✅ Voice synthesis available ({len(voices)} voices)")
        else:
            print("⚠️ Voice synthesis available but no voices")
        
        return True
        
    except ImportError:
        print("❌ pyttsx3 not available")
        return False
    except Exception as e:
        print(f"❌ Voice synthesis error: {e}")
        return False

def run_quick_test():
    """Run a quick functionality test"""
    
    try:
        # Add src to path
        sys.path.insert(0, 'src')
        
        from personality.config_manager import PersonalityConfigManager
        config_manager = PersonalityConfigManager()
        
        # Test config loading
        summary = config_manager.get_config_summary()
        print(f"✅ Configuration loads: {summary['personality_name']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False

def main():
    """Main setup function"""
    
    print("🆓 Free AI Trading Personality Setup for Windows")
    print("=" * 50)
    print("This script will help you set up the free AI personality system.")
    print()
    
    success = True
    
    # Step 1: Check Python
    print_step(1, "Checking Python Environment")
    if not check_python():
        success = False
    
    # Step 2: Install dependencies
    print_step(2, "Installing Python Dependencies")
    if success and not install_dependencies():
        success = False
    
    # Step 3: Setup directories
    print_step(3, "Setting Up Directories")
    if success:
        setup_directories()
    
    # Step 4: Setup configuration
    print_step(4, "Setting Up Configuration")
    if success and not setup_configuration():
        success = False
    
    # Step 5: Check Ollama
    print_step(5, "Checking Ollama")
    if success:
        check_ollama()  # Don't fail on Ollama issues
    
    # Step 6: Test voice
    print_step(6, "Testing Voice Synthesis")
    if success:
        test_voice()  # Don't fail on voice issues
    
    # Step 7: Quick test
    print_step(7, "Running Quick Test")
    if success and not run_quick_test():
        success = False
    
    # Show results
    print_header("Setup Results")
    
    if success:
        print("🎉 Setup completed successfully!")
        print()
        print("Next steps:")
        print("1. If using Ollama: ollama serve")
        print("2. Test setup: python test_free_personality.py")
        print("3. Run demo: python personality_demo.py")
        print("4. Read guide: FREE_SETUP_GUIDE_WINDOWS.md")
    else:
        print("❌ Setup encountered issues.")
        print()
        print("Troubleshooting:")
        print("1. Run: python troubleshoot_free_setup.py")
        print("2. Check: FREE_SETUP_GUIDE_WINDOWS.md")
        print("3. Try mock mode if Ollama isn't working")
    
    print()
    print("💡 Remember: This setup is completely free and runs locally!")

if __name__ == "__main__":
    main()