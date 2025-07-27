# main.py

import argparse
import logging
import shutil
import signal
import sys
from pathlib import Path

# Fix PyTorch 2.6+ weights loading issues
try:
    import torch
    import numpy
    # Add numpy globals to PyTorch safe globals list
    # Handle numpy namespace changes (newer versions use _core)
    try:
        torch.serialization.add_safe_globals([
            numpy._core.multiarray.scalar,
            numpy.dtype
        ])
    except AttributeError:
        # Fallback for older numpy versions
        torch.serialization.add_safe_globals([
            numpy.core.multiarray.scalar,
            numpy.dtype
        ])
except (ImportError, AttributeError):
    # Ignore if torch is not available or older version
    pass

from src.core.trading_system import TradingSystem

def check_system_dependencies():
    """
    Check system dependencies and provide clear feedback to users.
    
    Returns:
        bool: True if system can start, False if critical dependencies missing
    """
    try:
        from src.core.dependency_manager import check_dependencies
        
        print("Checking system dependencies...")
        results = check_dependencies()
        
        # Display dependency check results
        if not results['all_essential_available']:
            print("\n❌ CRITICAL: Missing essential dependencies!")
            print("The following dependencies are required for system operation:")
            for dep in results['missing_essential']:
                print(f"  - {dep}")
            print("\nPlease install missing dependencies using:")
            print("  pip install -r requirements.txt")
            print("\nOr install individually:")
            for dep in results['missing_essential']:
                print(f"  pip install {dep}")
            return False
        
        # Show warnings for missing enhanced dependencies
        if results['missing_enhanced']:
            print("\n⚠️  WARNING: Missing enhanced dependencies (reduced functionality):")
            for warning in results['warnings']:
                print(f"  - {warning}")
            print("For full functionality, consider installing:")
            for dep in results['missing_enhanced']:
                print(f"  pip install {dep}")
        
        # Show info for missing optional dependencies
        if results['missing_optional']:
            print("\nℹ️  INFO: Missing optional dependencies:")
            for dep in results['missing_optional']:
                print(f"  - {dep} (optional features disabled)")
        
        # Show system capabilities
        capabilities = results['system_capabilities']
        enabled_features = [k for k, v in capabilities.items() if v]
        disabled_features = [k for k, v in capabilities.items() if not v]
        
        if enabled_features:
            print(f"\n✅ Enabled features: {', '.join(enabled_features)}")
        if disabled_features:
            print(f"🚫 Disabled features: {', '.join(disabled_features)}")
        
        print("\n✅ Dependency check complete - system ready to start\n")
        return True
        
    except ImportError as e:
        print(f"⚠️  Warning: Could not import dependency manager: {e}")
        print("Proceeding with basic dependency assumptions...")
        return True
    except Exception as e:
        print(f"⚠️  Warning: Dependency check failed: {e}")
        print("Proceeding with basic dependency assumptions...")
        return True

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/trading.log'),
            logging.StreamHandler()
        ]
    )

def reset_workspace():
    import os
    import stat
    
    def handle_remove_readonly(func, path, exc):
        if os.path.exists(path):
            os.chmod(path, stat.S_IWRITE)
            func(path)
    
    dirs = ["models", "data", "logs"]
    for d in dirs:
        path = Path(d)
        if path.exists():
            print(f"Removing {d}/...")
            try:
                shutil.rmtree(path, onerror=handle_remove_readonly)
            except PermissionError as e:
                print(f"Warning: Could not remove {d}/ - {e}")
                # Try to clear contents instead
                for item in path.iterdir():
                    try:
                        if item.is_file():
                            item.chmod(0o777)
                            item.unlink()
                        elif item.is_dir():
                            shutil.rmtree(item, onerror=handle_remove_readonly)
                    except Exception as ex:
                        print(f"Warning: Could not remove {item} - {ex}")
        
        path.mkdir(parents=True, exist_ok=True)
        print(f"Created {d}/")
    
    print("Reset complete!")

def main():
    parser = argparse.ArgumentParser(description="Dopamine Trading System")
    parser.add_argument("--reset", action="store_true", help="Reset workspace")
    parser.add_argument("--skip-deps", action="store_true", help="Skip dependency checking")
    args = parser.parse_args()

    if args.reset:
        reset_workspace()
        print("Workspace reset complete")
        return

    # Ensure directories exist
    for d in ["models", "data", "logs"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    # Check dependencies unless skipped
    if not args.skip_deps:
        if not check_system_dependencies():
            print("\n❌ Cannot start system due to missing essential dependencies.")
            print("Use --skip-deps to bypass dependency checking (not recommended)")
            sys.exit(1)
    else:
        print("⚠️  Skipping dependency check as requested")

    setup_logging()
    
    system = TradingSystem()
    
    def shutdown_handler(signum, frame):
        # Create async wrapper for shutdown
        async def async_shutdown():
            await system.shutdown()
        
        import asyncio
        asyncio.run(async_shutdown())
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    try:
        # Use the unified trading system
        system.start()
    except Exception as e:
        logging.error(f"System failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()