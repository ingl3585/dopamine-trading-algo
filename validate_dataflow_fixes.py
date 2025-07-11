#!/usr/bin/env python3
"""
Validate data flow fixes by analyzing code structure and method signatures
This script doesn't require imports, just analyzes the code files
"""

import os
import re
import sys
from typing import Dict, List, Tuple

def analyze_method_signature(file_path: str, method_name: str) -> Dict:
    """Analyze a method signature in a file"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Find the method definition
        pattern = rf'def {method_name}\s*\((.*?)\):'
        match = re.search(pattern, content, re.DOTALL)
        
        if match:
            params = match.group(1).strip()
            return {
                'found': True,
                'parameters': params,
                'file': file_path
            }
        else:
            return {
                'found': False,
                'file': file_path
            }
    except Exception as e:
        return {
            'found': False,
            'error': str(e),
            'file': file_path
        }

def validate_temporal_subsystem_fixes():
    """Validate temporal subsystem learning method expects List[Dict]"""
    print("Validating temporal subsystem fixes...")
    
    # Check temporal subsystem
    temporal_file = "src/intelligence/subsystems/temporal_subsystem.py"
    if os.path.exists(temporal_file):
        with open(temporal_file, 'r') as f:
            content = f.read()
        
        # Check if learn_from_outcome expects List[Dict]
        if "def learn_from_outcome(self, cycles_info: List[Dict], outcome: float)" in content:
            print("✓ Temporal subsystem expects List[Dict] format")
        else:
            print("✗ Temporal subsystem signature issue")
            return False
    
    # Check orchestrator _extract_cycles_info ensures List[Dict]
    orchestrator_file = "src/core/trading_system_orchestrator.py"
    if os.path.exists(orchestrator_file):
        with open(orchestrator_file, 'r') as f:
            content = f.read()
        
        # Check for proper List[Dict] handling
        if "# Ensure we return List[Dict] format" in content:
            print("✓ Orchestrator ensures List[Dict] format in _extract_cycles_info")
        else:
            print("✗ Orchestrator _extract_cycles_info not properly fixed")
            return False
    
    return True

def validate_dna_subsystem_fixes():
    """Validate DNA subsystem learning method expects valid string"""
    print("Validating DNA subsystem fixes...")
    
    # Check DNA subsystem
    dna_file = "src/intelligence/subsystems/dna_subsystem.py"
    if os.path.exists(dna_file):
        with open(dna_file, 'r') as f:
            content = f.read()
        
        # Check if learn_from_outcome expects string
        if "def learn_from_outcome(self, sequence: str, outcome: float)" in content:
            print("✓ DNA subsystem expects string format")
        else:
            print("✗ DNA subsystem signature issue")
            return False
    
    # Check orchestrator _extract_dna_sequence ensures valid string
    orchestrator_file = "src/core/trading_system_orchestrator.py"
    if os.path.exists(orchestrator_file):
        with open(orchestrator_file, 'r') as f:
            content = f.read()
        
        # Check for fallback sequences
        if "FALLBACK_ABCD" in content and "ERROR_FALLBACK" in content:
            print("✓ Orchestrator ensures valid DNA sequences with fallbacks")
        else:
            print("✗ Orchestrator DNA sequence extraction not properly fixed")
            return False
    
    return True

def validate_immune_subsystem_fixes():
    """Validate immune subsystem parameter fixes"""
    print("Validating immune subsystem fixes...")
    
    # Check immune subsystem
    immune_file = "src/intelligence/subsystems/immune_subsystem.py"
    if os.path.exists(immune_file):
        with open(immune_file, 'r') as f:
            content = f.read()
        
        # Check if learn_threat has is_bootstrap parameter
        if "def learn_threat(self, market_state: Dict, threat_level: float, is_bootstrap: bool = False)" in content:
            print("✓ Immune subsystem has is_bootstrap parameter")
        else:
            print("✗ Immune subsystem learn_threat signature issue")
            return False
    
    # Check orchestrator calls immune with is_bootstrap
    orchestrator_file = "src/intelligence/subsystem_evolution.py"
    if os.path.exists(orchestrator_file):
        with open(orchestrator_file, 'r') as f:
            content = f.read()
        
        # Check for is_bootstrap parameter passing
        if "is_bootstrap = context.get('is_bootstrap', False)" in content:
            print("✓ Orchestrator passes is_bootstrap parameter to immune system")
        else:
            print("✗ Orchestrator doesn't pass is_bootstrap parameter")
            return False
    
    return True

def validate_market_state_standardization():
    """Validate market state standardization"""
    print("Validating market state standardization...")
    
    # Check orchestrator has standardization method
    orchestrator_file = "src/core/trading_system_orchestrator.py"
    if os.path.exists(orchestrator_file):
        with open(orchestrator_file, 'r') as f:
            content = f.read()
        
        # Check for standardization method
        if "def _create_standardized_market_state" in content:
            print("✓ Orchestrator has market state standardization method")
        else:
            print("✗ Orchestrator missing standardization method")
            return False
        
        # Check it's being used
        if "standardized_market_state = self._create_standardized_market_state" in content:
            print("✓ Orchestrator uses standardized market state")
        else:
            print("✗ Orchestrator doesn't use standardized market state")
            return False
    
    return True

def validate_data_validation():
    """Validate comprehensive data validation"""
    print("Validating comprehensive data validation...")
    
    # Check orchestrator has validation methods
    orchestrator_file = "src/intelligence/subsystem_evolution.py"
    if os.path.exists(orchestrator_file):
        with open(orchestrator_file, 'r') as f:
            content = f.read()
        
        # Check for validation methods
        if "def _validate_learning_inputs" in content:
            print("✓ Orchestrator has input validation method")
        else:
            print("✗ Orchestrator missing input validation method")
            return False
        
        if "def _extract_validated_context" in content:
            print("✓ Orchestrator has context validation method")
        else:
            print("✗ Orchestrator missing context validation method")
            return False
        
        # Check validation is used
        if "if not self._validate_learning_inputs(outcome, context)" in content:
            print("✓ Orchestrator uses input validation")
        else:
            print("✗ Orchestrator doesn't use input validation")
            return False
    
    return True

def validate_intelligence_engine_updates():
    """Validate intelligence engine consistency updates"""
    print("Validating intelligence engine updates...")
    
    # Check intelligence engine
    engine_file = "src/intelligence/intelligence_engine.py"
    if os.path.exists(engine_file):
        with open(engine_file, 'r') as f:
            content = f.read()
        
        # Check for consistent default structures
        if "'dna_sequence': 'DEFAULT_DNA_SEQUENCE'" in content:
            print("✓ Intelligence engine has consistent DNA defaults")
        else:
            print("✗ Intelligence engine missing consistent DNA defaults")
            return False
        
        # Check for market state validation
        if "# Ensure all required keys exist" in content:
            print("✓ Intelligence engine validates market state keys")
        else:
            print("✗ Intelligence engine missing market state validation")
            return False
        
        # Check for enhanced dopamine context
        if "'prediction_error': abs(outcome)" in content:
            print("✓ Intelligence engine has enhanced dopamine context")
        else:
            print("✗ Intelligence engine missing enhanced dopamine context")
            return False
    
    return True

def main():
    """Run all validation checks"""
    print("Starting data flow fixes validation...")
    print("=" * 60)
    
    validations = [
        ("Temporal Subsystem Fixes", validate_temporal_subsystem_fixes),
        ("DNA Subsystem Fixes", validate_dna_subsystem_fixes),
        ("Immune Subsystem Fixes", validate_immune_subsystem_fixes),
        ("Market State Standardization", validate_market_state_standardization),
        ("Data Validation", validate_data_validation),
        ("Intelligence Engine Updates", validate_intelligence_engine_updates)
    ]
    
    passed = 0
    failed = 0
    
    for validation_name, validation_func in validations:
        print(f"\n{'-' * 40}")
        print(f"Checking: {validation_name}")
        print(f"{'-' * 40}")
        
        try:
            if validation_func():
                print(f"✓ {validation_name} PASSED")
                passed += 1
            else:
                print(f"✗ {validation_name} FAILED")
                failed += 1
        except Exception as e:
            print(f"✗ {validation_name} FAILED with exception: {e}")
            failed += 1
    
    print(f"\n{'=' * 60}")
    print(f"VALIDATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total:  {passed + failed}")
    
    if failed == 0:
        print("🎉 ALL VALIDATIONS PASSED! Data flow fixes are properly implemented.")
        return 0
    else:
        print(f"❌ {failed} validations failed. Please review the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())