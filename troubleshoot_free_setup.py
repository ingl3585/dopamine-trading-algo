#!/usr/bin/env python3
"""
Free AI Trading Personality Troubleshooting Script

Diagnoses and fixes common issues with the free setup
"""

import asyncio
import sys
import subprocess
import os
import json
import logging
from typing import Dict, List, Tuple

# Suppress some logging for cleaner output
logging.getLogger('asyncio').setLevel(logging.WARNING)

class FreeTroubleshooter:
    """Troubleshoot free personality setup issues"""
    
    def __init__(self):
        self.issues_found = []
        self.fixes_applied = []
        
    async def run_diagnostics(self):
        """Run complete diagnostic suite"""
        
        print("🔍 AI Trading Personality Free Setup Troubleshooter")
        print("=" * 55)
        print()
        
        # Run all diagnostic checks
        await self.check_python_environment()
        await self.check_dependencies()
        await self.check_ollama_installation()
        await self.check_ollama_service()
        await self.check_models()
        await self.check_configuration()
        await self.check_file_structure()
        await self.check_voice_synthesis()
        await self.test_basic_functionality()
        
        # Show results and suggestions
        self.show_diagnostic_results()
    
    async def check_python_environment(self):
        """Check Python environment"""
        
        print("🐍 Checking Python Environment")
        print("-" * 30)
        
        # Check Python version
        version = sys.version_info
        if version.major >= 3 and version.minor >= 8:
            print(f"✅ Python {version.major}.{version.minor}.{version.micro} (Good)")
        else:
            print(f"❌ Python {version.major}.{version.minor}.{version.micro} (Need 3.8+)")
            self.issues_found.append("Python version too old")
        
        # Check if we can import sys modules
        try:
            import asyncio, json, os
            print("✅ Core Python modules available")
        except ImportError as e:
            print(f"❌ Core Python modules missing: {e}")
            self.issues_found.append(f"Python module error: {e}")
        
        print()
    
    async def check_dependencies(self):
        """Check required Python packages"""
        
        print("📦 Checking Python Dependencies")
        print("-" * 32)
        
        required_packages = [
            ('numpy', 'numpy'),
            ('scipy', 'scipy'),
            ('pandas', 'pandas'),
            ('rich', 'rich'),
            ('aiohttp', 'aiohttp'),
            ('pyttsx3', 'pyttsx3')
        ]
        
        missing_packages = []
        
        for package_name, import_name in required_packages:
            try:
                __import__(import_name)
                print(f"✅ {package_name}")
            except ImportError:
                print(f"❌ {package_name} (missing)")
                missing_packages.append(package_name)
        
        if missing_packages:
            self.issues_found.append(f"Missing packages: {', '.join(missing_packages)}")
            print(f"\n💡 To fix: pip install {' '.join(missing_packages)}")
        
        print()
    
    async def check_ollama_installation(self):
        """Check Ollama installation"""
        
        print("🦙 Checking Ollama Installation")
        print("-" * 30)
        
        try:
            # Check if ollama command exists
            result = subprocess.run(['ollama', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                version = result.stdout.strip()
                print(f"✅ Ollama installed: {version}")
            else:
                print("❌ Ollama command failed")
                self.issues_found.append("Ollama installation issue")
                
        except subprocess.TimeoutExpired:
            print("❌ Ollama command timeout")
            self.issues_found.append("Ollama not responding")
        except FileNotFoundError:
            print("❌ Ollama not found")
            self.issues_found.append("Ollama not installed")
            print("💡 Download from: https://ollama.ai/download")
        except Exception as e:
            print(f"❌ Ollama check error: {e}")
            self.issues_found.append(f"Ollama error: {e}")
        
        print()
    
    async def check_ollama_service(self):
        """Check if Ollama service is running"""
        
        print("🔄 Checking Ollama Service")
        print("-" * 25)
        
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.get('http://localhost:11434/api/version', timeout=5) as response:
                        if response.status == 200:
                            data = await response.json()
                            print(f"✅ Ollama service running: {data.get('version', 'unknown')}")
                        else:
                            print(f"❌ Ollama service error: HTTP {response.status}")
                            self.issues_found.append("Ollama service not responding properly")
                except asyncio.TimeoutError:
                    print("❌ Ollama service timeout")
                    self.issues_found.append("Ollama service not responding")
                    print("💡 Start with: ollama serve")
                except aiohttp.ClientConnectorError:
                    print("❌ Ollama service not running")
                    self.issues_found.append("Ollama service not running")
                    print("💡 Start with: ollama serve")
                    
        except ImportError:
            print("❌ aiohttp not available for service check")
            
        print()
    
    async def check_models(self):
        """Check available Ollama models"""
        
        print("🤖 Checking Ollama Models")
        print("-" * 23)
        
        try:
            result = subprocess.run(['ollama', 'list'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                output = result.stdout.strip()
                if 'llama2' in output.lower():
                    print("✅ Llama2 model found")
                elif output and len(output.split('\n')) > 1:
                    print("✅ Some models available")
                    print(f"   Models: {output.split()[0] if output.split() else 'unknown'}")
                else:
                    print("❌ No models installed")
                    self.issues_found.append("No Ollama models installed")
                    print("💡 Install with: ollama pull llama2:7b-chat")
            else:
                print("❌ Cannot list models")
                self.issues_found.append("Cannot access Ollama models")
                
        except Exception as e:
            print(f"❌ Model check error: {e}")
            
        print()
    
    async def check_configuration(self):
        """Check configuration files"""
        
        print("⚙️ Checking Configuration")
        print("-" * 24)
        
        config_files = [
            'config/personality_config.json',
            'config/personality_config_free.json',
            'config/personality_config_mock.json'
        ]
        
        config_found = False
        
        for config_file in config_files:
            if os.path.exists(config_file):
                print(f"✅ {config_file}")
                config_found = True
                
                # Try to parse JSON
                try:
                    with open(config_file, 'r') as f:
                        config_data = json.load(f)
                    
                    # Check key sections
                    if 'personality' in config_data:
                        print(f"   Personality: {config_data['personality'].get('personality_name', 'unknown')}")
                    if 'llm' in config_data:
                        print(f"   LLM Model: {config_data['llm'].get('model_name', 'unknown')}")
                    
                except json.JSONDecodeError as e:
                    print(f"   ❌ Invalid JSON: {e}")
                    self.issues_found.append(f"Invalid config JSON in {config_file}")
                except Exception as e:
                    print(f"   ❌ Config error: {e}")
            else:
                print(f"❌ {config_file} (missing)")
        
        if not config_found:
            self.issues_found.append("No configuration files found")
        
        print()
    
    async def check_file_structure(self):
        """Check required file structure"""
        
        print("📁 Checking File Structure")
        print("-" * 25)
        
        required_dirs = [
            'src',
            'src/personality',
            'config',
            'data'
        ]
        
        required_files = [
            'src/personality/__init__.py',
            'src/personality/trading_personality.py',
            'src/personality/llm_client.py',
            'src/personality/emotional_engine.py'
        ]
        
        for directory in required_dirs:
            if os.path.exists(directory):
                print(f"✅ {directory}/")
            else:
                print(f"❌ {directory}/ (missing)")
                self.issues_found.append(f"Missing directory: {directory}")
        
        for file_path in required_files:
            if os.path.exists(file_path):
                print(f"✅ {file_path}")
            else:
                print(f"❌ {file_path} (missing)")
                self.issues_found.append(f"Missing file: {file_path}")
        
        print()
    
    async def check_voice_synthesis(self):
        """Check voice synthesis capability"""
        
        print("🔊 Checking Voice Synthesis")
        print("-" * 26)
        
        try:
            import pyttsx3
            engine = pyttsx3.init()
            
            # Test basic properties
            voices = engine.getProperty('voices')
            if voices:
                print(f"✅ pyttsx3 available ({len(voices)} voices)")
                # Find English voices
                english_voices = [v for v in voices if 'english' in v.name.lower()]
                if english_voices:
                    print(f"   English voices: {len(english_voices)}")
                else:
                    print("   ⚠️ No English voices found")
            else:
                print("⚠️ pyttsx3 available but no voices")
                
        except ImportError:
            print("❌ pyttsx3 not available")
            self.issues_found.append("Voice synthesis not available")
            print("💡 Install with: pip install pyttsx3")
        except Exception as e:
            print(f"❌ Voice synthesis error: {e}")
            self.issues_found.append(f"Voice synthesis error: {e}")
        
        print()
    
    async def test_basic_functionality(self):
        """Test basic personality functionality"""
        
        print("🧪 Testing Basic Functionality")
        print("-" * 30)
        
        try:
            # Add src to path
            sys.path.insert(0, 'src')
            
            # Test imports
            from personality.trading_personality import TradingPersonality, PersonalityConfig
            from personality.config_manager import PersonalityConfigManager
            print("✅ Personality modules import successfully")
            
            # Test configuration
            config_manager = PersonalityConfigManager()
            config = config_manager.get_personality_config()
            print("✅ Configuration loads successfully")
            
            # Test personality creation
            personality = TradingPersonality(config)
            print("✅ Personality creates successfully")
            
            # Test emotional state
            emotional_state = personality.get_current_emotional_state()
            print(f"✅ Emotional state: {emotional_state.primary_emotion.value}")
            
            # Cleanup
            personality.shutdown()
            
        except Exception as e:
            print(f"❌ Basic functionality test failed: {e}")
            self.issues_found.append(f"Basic functionality error: {e}")
        
        print()
    
    def show_diagnostic_results(self):
        """Show diagnostic results and recommendations"""
        
        print("📋 Diagnostic Results")
        print("=" * 20)
        
        if not self.issues_found:
            print("🎉 ALL CHECKS PASSED!")
            print("Your free AI Trading Personality setup appears to be working correctly.")
            print("\nNext steps:")
            print("1. Run: python test_free_personality.py")
            print("2. Run: python personality_demo.py")
            print("3. Read: FREE_SETUP_GUIDE_WINDOWS.md")
        else:
            print(f"❌ Found {len(self.issues_found)} issues:")
            print()
            
            for i, issue in enumerate(self.issues_found, 1):
                print(f"{i}. {issue}")
            
            print("\n🔧 Recommended fixes:")
            print()
            
            # Provide specific fixes
            if any("Missing packages" in issue for issue in self.issues_found):
                missing = [issue.split(": ")[1] for issue in self.issues_found if "Missing packages" in issue][0]
                print(f"📦 Install packages: pip install {missing}")
            
            if any("Ollama not installed" in issue for issue in self.issues_found):
                print("🦙 Install Ollama: https://ollama.ai/download")
            
            if any("Ollama service not running" in issue for issue in self.issues_found):
                print("🔄 Start Ollama: ollama serve")
            
            if any("No Ollama models" in issue for issue in self.issues_found):
                print("🤖 Install model: ollama pull llama2:7b-chat")
            
            if any("Voice synthesis not available" in issue for issue in self.issues_found):
                print("🔊 Install voice: pip install pyttsx3")
            
            if any("configuration" in issue.lower() for issue in self.issues_found):
                print("⚙️ Copy config: copy config\\personality_config_free.json config\\personality_config.json")
            
            print("\n🆘 If issues persist:")
            print("1. Try mock mode: copy config\\personality_config_mock.json config\\personality_config.json")
            print("2. Check FREE_SETUP_GUIDE_WINDOWS.md for detailed troubleshooting")
            print("3. Run this script again after applying fixes")

async def main():
    """Main troubleshooting function"""
    
    troubleshooter = FreeTroubleshooter()
    await troubleshooter.run_diagnostics()

if __name__ == "__main__":
    asyncio.run(main())