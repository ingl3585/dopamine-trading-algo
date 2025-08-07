"""
State File Manager - Safe file discovery, validation, and cleanup operations

This module provides secure file operations for state cleanup, including:
- Safe file discovery and categorization
- File validation and integrity checks
- Atomic file operations with rollback capabilities
- Orphaned file detection and cleanup
- Thread-safe operations

Key Features:
- Comprehensive file metadata collection
- Safe deletion with backup capabilities
- Transaction-like operations for bulk changes
- File integrity validation
- Automatic orphaned file detection
"""

import logging
import os
import shutil
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
import json
import re

from .retention_policies import FileInfo, FilePriority

logger = logging.getLogger(__name__)


class FileOperationError(Exception):
    """Exception raised for file operation errors"""
    pass


class FileValidationError(Exception):
    """Exception raised for file validation errors"""
    pass


class StateFileManager:
    """
    Manages state file operations with safety and transaction capabilities
    
    This class provides high-level operations for discovering, analyzing,
    and safely manipulating state files while maintaining data integrity.
    """
    
    def __init__(self, base_path: Union[str, Path]):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        
        self._operation_lock = threading.RLock()
        self._backup_dir = self.base_path / ".cleanup_backups"
        self._temp_dir = self.base_path / ".cleanup_temp"
        
        # File patterns for categorization
        self._patterns = {
            'state_file': re.compile(r'^system_state_\d{8}_\d{6}\.json$'),
            'temp_file': re.compile(r'^.*\.tmp$'),
            'backup_file': re.compile(r'^.*\.backup$'),
            'transaction_file': re.compile(r'^transaction_.*\.json$'),
        }
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize cleanup directories
        self._ensure_cleanup_directories()
    
    def _ensure_cleanup_directories(self):
        """Ensure cleanup working directories exist"""
        try:
            self._backup_dir.mkdir(exist_ok=True)
            self._temp_dir.mkdir(exist_ok=True)
            
            # Create .gitignore for cleanup directories
            gitignore_path = self._backup_dir / ".gitignore"
            if not gitignore_path.exists():
                gitignore_path.write_text("*\n!.gitignore\n")
                
            gitignore_path = self._temp_dir / ".gitignore" 
            if not gitignore_path.exists():
                gitignore_path.write_text("*\n!.gitignore\n")
                
        except Exception as e:
            self.logger.warning(f"Failed to create cleanup directories: {e}")
    
    def discover_state_files(self, include_temp: bool = True) -> List[FileInfo]:
        """
        Discover all state files in the base directory
        
        Args:
            include_temp: Whether to include temporary files
            
        Returns:
            List of FileInfo objects for discovered files
        """
        with self._operation_lock:
            files = []
            
            try:
                # Find all relevant files
                for file_path in self.base_path.iterdir():
                    if not file_path.is_file():
                        continue
                    
                    # Skip cleanup directories
                    if file_path.parent.name in ['.cleanup_backups', '.cleanup_temp']:
                        continue
                    
                    # Categorize file
                    file_info = self._analyze_file(file_path)
                    
                    if file_info is None:
                        continue
                    
                    # Filter temp files if requested
                    if not include_temp and file_info.is_temp:
                        continue
                    
                    files.append(file_info)
                
                self.logger.info(f"Discovered {len(files)} state files")
                return files
                
            except Exception as e:
                self.logger.error(f"Error discovering state files: {e}")
                raise FileOperationError(f"Failed to discover state files: {e}")
    
    def _analyze_file(self, file_path: Path) -> Optional[FileInfo]:
        """
        Analyze a file and create FileInfo object
        
        Args:
            file_path: Path to the file to analyze
            
        Returns:
            FileInfo object or None if file should be ignored
        """
        try:
            stat = file_path.stat()
            filename = file_path.name
            
            # Determine file type and priority
            priority = self._determine_file_priority(file_path)
            if priority is None:
                return None  # Skip files we don't care about
            
            # Check for various file characteristics
            is_temp = self._patterns['temp_file'].match(filename) is not None
            is_transaction = self._patterns['transaction_file'].match(filename) is not None
            is_orphaned = self._is_orphaned_file(file_path)
            
            return FileInfo(
                path=file_path,
                size_bytes=stat.st_size,
                created_time=stat.st_ctime,
                modified_time=stat.st_mtime,
                priority=priority,
                is_temp=is_temp,
                is_transaction=is_transaction,
                is_orphaned=is_orphaned
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to analyze file {file_path}: {e}")
            return None
    
    def _determine_file_priority(self, file_path: Path) -> Optional[FilePriority]:
        """
        Determine the priority level of a file based on its characteristics
        
        Args:
            file_path: Path to the file
            
        Returns:
            FilePriority or None if file should be ignored
        """
        filename = file_path.name
        
        # Skip non-state files
        if not (self._patterns['state_file'].match(filename) or
                self._patterns['temp_file'].match(filename) or
                self._patterns['backup_file'].match(filename) or
                self._patterns['transaction_file'].match(filename)):
            return None
        
        # Critical files (recent state files, active transactions)
        if self._patterns['transaction_file'].match(filename):
            return FilePriority.CRITICAL
        
        # Check if file is very recent (last hour)
        try:
            stat = file_path.stat()
            age_hours = (time.time() - stat.st_mtime) / 3600
            
            if age_hours < 1:
                return FilePriority.CRITICAL
            elif age_hours < 24:
                return FilePriority.HIGH
            elif age_hours < 168:  # 1 week
                return FilePriority.MEDIUM
            else:
                return FilePriority.LOW
                
        except Exception:
            # Default to medium if we can't determine age
            return FilePriority.MEDIUM
    
    def _is_orphaned_file(self, file_path: Path) -> bool:
        """
        Check if a file appears to be orphaned (e.g., .tmp file without corresponding main file)
        
        Args:
            file_path: Path to check
            
        Returns:
            True if file appears to be orphaned
        """
        filename = file_path.name
        
        # Check for orphaned .tmp files
        if filename.endswith('.tmp'):
            # Look for corresponding non-temp file
            expected_main_file = file_path.with_suffix('')
            if not expected_main_file.exists():
                # Check if the temp file is old (> 1 hour)
                try:
                    stat = file_path.stat()
                    age_hours = (time.time() - stat.st_mtime) / 3600
                    return age_hours > 1  # Consider temp files orphaned after 1 hour
                except Exception:
                    return True
        
        return False
    
    def validate_file(self, file_info: FileInfo) -> bool:
        """
        Validate a state file's integrity
        
        Args:
            file_info: File to validate
            
        Returns:
            True if file is valid, False otherwise
        """
        try:
            # Skip validation for temp files (they may be incomplete)
            if file_info.is_temp:
                return True
            
            # For JSON files, try to parse them
            if file_info.path.suffix == '.json':
                with open(file_info.path, 'r') as f:
                    data = json.load(f)
                
                # Basic structure validation for state files
                if self._patterns['state_file'].match(file_info.path.name):
                    required_fields = ['timestamp', 'version', 'components', 'metadata']
                    if not all(field in data for field in required_fields):
                        self.logger.warning(f"Invalid state file structure: {file_info.path.name}")
                        return False
            
            # File exists and is readable
            if not file_info.path.exists() or not file_info.path.is_file():
                return False
            
            # File has reasonable size (not empty, not too large)
            if file_info.size_bytes == 0:
                self.logger.warning(f"Empty file detected: {file_info.path.name}")
                return False
            
            if file_info.size_bytes > 100 * 1024 * 1024:  # 100MB limit
                self.logger.warning(f"Unusually large file: {file_info.path.name} ({file_info.size_bytes} bytes)")
                return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"File validation failed for {file_info.path.name}: {e}")
            return False
    
    def get_total_size(self, files: List[FileInfo]) -> int:
        """
        Calculate total size of a list of files
        
        Args:
            files: List of FileInfo objects
            
        Returns:
            Total size in bytes
        """
        return sum(f.size_bytes for f in files)
    
    def get_size_breakdown(self, files: List[FileInfo]) -> Dict[str, int]:
        """
        Get size breakdown by file type and priority
        
        Args:
            files: List of FileInfo objects
            
        Returns:
            Dictionary with size breakdowns
        """
        breakdown = {
            'total_size': 0,
            'by_priority': {p.value: 0 for p in FilePriority},
            'temp_files': 0,
            'transaction_files': 0,
            'orphaned_files': 0
        }
        
        for file_info in files:
            size = file_info.size_bytes
            breakdown['total_size'] += size
            breakdown['by_priority'][file_info.priority.value] += size
            
            if file_info.is_temp:
                breakdown['temp_files'] += size
            if file_info.is_transaction:
                breakdown['transaction_files'] += size
            if file_info.is_orphaned:
                breakdown['orphaned_files'] += size
        
        return breakdown
    
    @contextmanager
    def cleanup_transaction(self, transaction_id: str):
        """
        Context manager for safe cleanup operations with rollback capability
        
        Args:
            transaction_id: Unique identifier for this cleanup transaction
        """
        backup_path = self._backup_dir / f"cleanup_{transaction_id}"
        backup_path.mkdir(exist_ok=True)
        
        backed_up_files = []
        
        try:
            self.logger.info(f"Starting cleanup transaction: {transaction_id}")
            
            # Provide transaction context
            yield CleanupTransaction(self, transaction_id, backup_path, backed_up_files)
            
            self.logger.info(f"Cleanup transaction {transaction_id} completed successfully")
            
            # Clean up backup if successful
            try:
                shutil.rmtree(backup_path)
            except Exception as e:
                self.logger.warning(f"Failed to cleanup backup directory: {e}")
                
        except Exception as e:
            self.logger.error(f"Cleanup transaction {transaction_id} failed: {e}")
            
            # Attempt rollback
            try:
                self._rollback_transaction(backup_path, backed_up_files)
                self.logger.info(f"Successfully rolled back transaction {transaction_id}")
            except Exception as rollback_error:
                self.logger.error(f"Rollback failed for transaction {transaction_id}: {rollback_error}")
                raise FileOperationError(f"Transaction failed and rollback failed: {rollback_error}")
            
            raise
    
    def _rollback_transaction(self, backup_path: Path, backed_up_files: List[Tuple[Path, Path]]):
        """
        Rollback a failed cleanup transaction
        
        Args:
            backup_path: Path to backup directory
            backed_up_files: List of (original_path, backup_path) tuples
        """
        for original_path, backup_file_path in backed_up_files:
            try:
                if backup_file_path.exists():
                    shutil.copy2(backup_file_path, original_path)
                    self.logger.debug(f"Restored {original_path} from backup")
            except Exception as e:
                self.logger.error(f"Failed to restore {original_path}: {e}")
                raise
    
    def delete_files_safely(self, files_to_delete: List[FileInfo], backup_path: Path) -> List[Tuple[Path, Path]]:
        """
        Safely delete files with backup
        
        Args:
            files_to_delete: List of files to delete
            backup_path: Directory to store backups
            
        Returns:
            List of (original_path, backup_path) tuples for rollback
        """
        backed_up_files = []
        
        try:
            # First, backup all files
            for file_info in files_to_delete:
                if not file_info.path.exists():
                    continue
                
                # Create backup
                backup_file_path = backup_path / file_info.path.name
                shutil.copy2(file_info.path, backup_file_path)
                backed_up_files.append((file_info.path, backup_file_path))
                
                self.logger.debug(f"Backed up {file_info.path.name}")
            
            # Then delete original files
            deleted_count = 0
            total_size_deleted = 0
            
            for file_info in files_to_delete:
                if not file_info.path.exists():
                    continue
                
                try:
                    total_size_deleted += file_info.size_bytes
                    file_info.path.unlink()
                    deleted_count += 1
                    self.logger.debug(f"Deleted {file_info.path.name}")
                except Exception as e:
                    self.logger.error(f"Failed to delete {file_info.path.name}: {e}")
                    raise
            
            self.logger.info(f"Successfully deleted {deleted_count} files "
                           f"({total_size_deleted / (1024*1024):.1f}MB)")
            
            return backed_up_files
            
        except Exception as e:
            self.logger.error(f"Error during file deletion: {e}")
            raise
    
    def cleanup_orphaned_temp_files(self) -> int:
        """
        Clean up orphaned temporary files
        
        Returns:
            Number of files cleaned up
        """
        with self._operation_lock:
            try:
                files = self.discover_state_files(include_temp=True)
                orphaned_files = [f for f in files if f.is_orphaned and f.is_temp]
                
                if not orphaned_files:
                    self.logger.debug("No orphaned temp files found")
                    return 0
                
                self.logger.info(f"Found {len(orphaned_files)} orphaned temp files")
                
                # Delete orphaned temp files directly (no backup needed for temp files)
                deleted_count = 0
                for file_info in orphaned_files:
                    try:
                        if file_info.path.exists():
                            file_info.path.unlink()
                            deleted_count += 1
                            self.logger.debug(f"Deleted orphaned temp file: {file_info.path.name}")
                    except Exception as e:
                        self.logger.warning(f"Failed to delete orphaned temp file {file_info.path.name}: {e}")
                
                self.logger.info(f"Cleaned up {deleted_count} orphaned temp files")
                return deleted_count
                
            except Exception as e:
                self.logger.error(f"Error during orphaned temp file cleanup: {e}")
                raise FileOperationError(f"Failed to cleanup orphaned temp files: {e}")
    
    def get_manager_status(self) -> Dict[str, Union[str, int, float]]:
        """
        Get current status of the file manager
        
        Returns:
            Dictionary with status information
        """
        with self._operation_lock:
            try:
                files = self.discover_state_files(include_temp=True)
                size_breakdown = self.get_size_breakdown(files)
                
                return {
                    'base_path': str(self.base_path),
                    'total_files': len(files),
                    'total_size_mb': size_breakdown['total_size'] / (1024 * 1024),
                    'temp_files': sum(1 for f in files if f.is_temp),
                    'orphaned_files': sum(1 for f in files if f.is_orphaned),
                    'transaction_files': sum(1 for f in files if f.is_transaction),
                    'critical_files': sum(1 for f in files if f.priority == FilePriority.CRITICAL),
                    'high_priority_files': sum(1 for f in files if f.priority == FilePriority.HIGH),
                    'medium_priority_files': sum(1 for f in files if f.priority == FilePriority.MEDIUM),
                    'low_priority_files': sum(1 for f in files if f.priority == FilePriority.LOW),
                    'backup_dir_exists': self._backup_dir.exists(),
                    'temp_dir_exists': self._temp_dir.exists()
                }
                
            except Exception as e:
                self.logger.error(f"Error getting manager status: {e}")
                return {'error': str(e)}


class CleanupTransaction:
    """
    Represents an active cleanup transaction with rollback capabilities
    """
    
    def __init__(self, manager: StateFileManager, transaction_id: str, 
                 backup_path: Path, backed_up_files: List):
        self.manager = manager
        self.transaction_id = transaction_id
        self.backup_path = backup_path
        self.backed_up_files = backed_up_files
        self.logger = logging.getLogger(f"{__name__}.CleanupTransaction")
    
    def delete_files(self, files_to_delete: List[FileInfo]):
        """
        Delete files as part of this transaction
        
        Args:
            files_to_delete: List of files to delete
        """
        if not files_to_delete:
            return
        
        self.logger.info(f"Deleting {len(files_to_delete)} files in transaction {self.transaction_id}")
        
        # Perform the deletion with backup
        new_backups = self.manager.delete_files_safely(files_to_delete, self.backup_path)
        self.backed_up_files.extend(new_backups)
    
    def get_transaction_info(self) -> Dict[str, Union[str, int]]:
        """
        Get information about this transaction
        
        Returns:
            Dictionary with transaction information
        """
        return {
            'transaction_id': self.transaction_id,
            'backup_path': str(self.backup_path),
            'backed_up_files_count': len(self.backed_up_files),
            'backup_exists': self.backup_path.exists()
        }