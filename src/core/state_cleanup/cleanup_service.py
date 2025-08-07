"""
Cleanup Service - Orchestrates comprehensive state file cleanup operations

This module provides the main cleanup orchestration logic that combines
file discovery, retention policy evaluation, and safe cleanup execution.

Key Features:
- Comprehensive cleanup planning and execution
- Multi-phase cleanup with safety checks
- Detailed reporting and logging
- Integration with retention policies and file manager
- Automatic scheduling and manual triggers
- Performance monitoring and optimization

Cleanup Phases:
1. Discovery: Find and categorize all state files
2. Analysis: Apply retention policies and create cleanup plan
3. Validation: Verify cleanup plan safety and integrity
4. Execution: Perform cleanup with transaction safety
5. Reporting: Generate cleanup report and metrics
"""

import logging
import time
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

from .file_manager import StateFileManager, FileOperationError
from .retention_policies import RetentionPolicySet, FileInfo, FilePriority

logger = logging.getLogger(__name__)


@dataclass
class CleanupPlan:
    """Represents a planned cleanup operation"""
    plan_id: str
    created_at: float
    files_to_delete: List[FileInfo]
    files_to_keep: List[FileInfo]
    total_files_found: int
    size_to_delete_mb: float
    size_to_keep_mb: float
    estimated_cleanup_time_seconds: float
    safety_warnings: List[str]
    policy_decisions: Dict[str, int]  # policy_name -> files_deleted_count
    
    @property
    def deletion_percentage(self) -> float:
        """Percentage of files that will be deleted"""
        if self.total_files_found == 0:
            return 0.0
        return (len(self.files_to_delete) / self.total_files_found) * 100
    
    @property
    def space_savings_mb(self) -> float:
        """Amount of space that will be freed in MB"""
        return self.size_to_delete_mb
    
    def is_safe_to_execute(self) -> bool:
        """Check if the cleanup plan is safe to execute"""
        # Don't delete more than 80% of files in one operation
        if self.deletion_percentage > 80:
            return False
        
        # Don't delete if there are critical safety warnings
        critical_warnings = [w for w in self.safety_warnings if 'CRITICAL' in w.upper()]
        if critical_warnings:
            return False
        
        # Ensure we keep some files
        if len(self.files_to_keep) == 0:
            return False
        
        return True


@dataclass 
class CleanupResult:
    """Results of a cleanup operation"""
    plan_id: str
    started_at: float
    completed_at: float
    success: bool
    files_deleted: int
    files_failed_to_delete: int
    size_freed_mb: float
    errors: List[str]
    warnings: List[str]
    transaction_id: Optional[str] = None
    rollback_performed: bool = False
    
    @property
    def duration_seconds(self) -> float:
        """Total execution time in seconds"""
        return self.completed_at - self.started_at
    
    @property
    def success_rate(self) -> float:
        """Percentage of files successfully deleted"""
        total_attempted = self.files_deleted + self.files_failed_to_delete
        if total_attempted == 0:
            return 100.0
        return (self.files_deleted / total_attempted) * 100


class CleanupService:
    """
    Main cleanup service that orchestrates all cleanup operations
    
    This service combines file discovery, retention policy evaluation,
    and safe cleanup execution to provide comprehensive state file management.
    """
    
    def __init__(self, 
                 file_manager: StateFileManager,
                 retention_policies: RetentionPolicySet,
                 max_files_per_cleanup: int = 1000,
                 max_size_per_cleanup_mb: float = 1000.0):
        self.file_manager = file_manager
        self.retention_policies = retention_policies
        self.max_files_per_cleanup = max_files_per_cleanup
        self.max_size_per_cleanup_mb = max_size_per_cleanup_mb
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Statistics tracking
        self._cleanup_history: List[CleanupResult] = []
        self._last_cleanup_time: Optional[float] = None
        self._total_files_deleted = 0
        self._total_space_freed_mb = 0.0
    
    def create_cleanup_plan(self, dry_run: bool = True) -> CleanupPlan:
        """
        Create a comprehensive cleanup plan based on current retention policies
        
        Args:
            dry_run: If True, only plan without executing
            
        Returns:
            CleanupPlan with detailed cleanup strategy
        """
        plan_id = str(uuid.uuid4())[:8]
        start_time = time.time()
        
        self.logger.info(f"Creating cleanup plan {plan_id} (dry_run={dry_run})")
        
        try:
            # Phase 1: Discovery
            self.logger.debug("Phase 1: Discovering state files")
            all_files = self.file_manager.discover_state_files(include_temp=True)
            
            if not all_files:
                self.logger.info("No state files found for cleanup")
                return CleanupPlan(
                    plan_id=plan_id,
                    created_at=start_time,
                    files_to_delete=[],
                    files_to_keep=[],
                    total_files_found=0,
                    size_to_delete_mb=0.0,
                    size_to_keep_mb=0.0,
                    estimated_cleanup_time_seconds=0.0,
                    safety_warnings=[],
                    policy_decisions={}
                )
            
            self.logger.info(f"Found {len(all_files)} total files for evaluation")
            
            # Phase 2: Policy Evaluation
            self.logger.debug("Phase 2: Applying retention policies")
            retention_decisions = self.retention_policies.evaluate_retention(all_files)
            
            # Separate files to keep vs delete
            files_to_keep = []
            files_to_delete = []
            
            for file_info in all_files:
                if retention_decisions.get(file_info.path, True):
                    files_to_keep.append(file_info)
                else:
                    files_to_delete.append(file_info)
            
            self.logger.info(f"Retention decision: keep {len(files_to_keep)}, delete {len(files_to_delete)}")
            
            # Phase 3: Safety Analysis
            self.logger.debug("Phase 3: Performing safety analysis")
            safety_warnings = self._analyze_cleanup_safety(files_to_delete, files_to_keep)
            
            # Apply safety limits
            files_to_delete = self._apply_safety_limits(files_to_delete)
            
            # Calculate sizes and estimates
            size_to_delete = sum(f.size_bytes for f in files_to_delete) / (1024 * 1024)
            size_to_keep = sum(f.size_bytes for f in files_to_keep) / (1024 * 1024)
            estimated_time = self._estimate_cleanup_time(files_to_delete)
            
            # Generate policy decision summary
            policy_decisions = self._summarize_policy_decisions(all_files, retention_decisions)
            
            plan = CleanupPlan(
                plan_id=plan_id,
                created_at=start_time,
                files_to_delete=files_to_delete,
                files_to_keep=files_to_keep,
                total_files_found=len(all_files),
                size_to_delete_mb=size_to_delete,
                size_to_keep_mb=size_to_keep,
                estimated_cleanup_time_seconds=estimated_time,
                safety_warnings=safety_warnings,
                policy_decisions=policy_decisions
            )
            
            self.logger.info(f"Cleanup plan {plan_id} created: "
                           f"{len(files_to_delete)} files to delete ({size_to_delete:.1f}MB), "
                           f"{len(safety_warnings)} warnings")
            
            return plan
            
        except Exception as e:
            self.logger.error(f"Failed to create cleanup plan {plan_id}: {e}")
            raise
    
    def execute_cleanup_plan(self, plan: CleanupPlan) -> CleanupResult:
        """
        Execute a cleanup plan with full transaction safety
        
        Args:
            plan: CleanupPlan to execute
            
        Returns:
            CleanupResult with execution details
        """
        start_time = time.time()
        transaction_id = str(uuid.uuid4())[:8]
        
        self.logger.info(f"Executing cleanup plan {plan.plan_id} with transaction {transaction_id}")
        
        # Pre-execution validation
        if not plan.is_safe_to_execute():
            error_msg = f"Cleanup plan {plan.plan_id} failed safety validation"
            self.logger.error(error_msg)
            return CleanupResult(
                plan_id=plan.plan_id,
                started_at=start_time,
                completed_at=time.time(),
                success=False,
                files_deleted=0,
                files_failed_to_delete=0,
                size_freed_mb=0.0,
                errors=[error_msg],
                warnings=plan.safety_warnings
            )
        
        errors = []
        warnings = list(plan.safety_warnings)
        files_deleted = 0
        files_failed = 0
        size_freed = 0.0
        rollback_performed = False
        
        # Execute with transaction safety
        try:
            with self.file_manager.cleanup_transaction(transaction_id) as transaction:
                self.logger.info(f"Starting cleanup transaction {transaction_id}")
                
                # Delete files in batches for better progress tracking
                batch_size = 50
                files_to_delete = plan.files_to_delete
                
                for i in range(0, len(files_to_delete), batch_size):
                    batch = files_to_delete[i:i + batch_size]
                    batch_num = (i // batch_size) + 1
                    total_batches = (len(files_to_delete) + batch_size - 1) // batch_size
                    
                    self.logger.info(f"Processing batch {batch_num}/{total_batches} "
                                   f"({len(batch)} files)")
                    
                    try:
                        # Pre-validate batch files
                        valid_files = []
                        for file_info in batch:
                            if file_info.path.exists():
                                if self.file_manager.validate_file(file_info):
                                    valid_files.append(file_info)
                                else:
                                    warnings.append(f"Skipping invalid file: {file_info.path.name}")
                            else:
                                warnings.append(f"File no longer exists: {file_info.path.name}")
                        
                        if valid_files:
                            # Track files before deletion for size calculation
                            batch_size_bytes = sum(f.size_bytes for f in valid_files)
                            
                            # Perform deletion
                            transaction.delete_files(valid_files)
                            
                            # Update counters
                            files_deleted += len(valid_files)
                            size_freed += batch_size_bytes / (1024 * 1024)
                            
                            self.logger.debug(f"Batch {batch_num} completed: "
                                            f"deleted {len(valid_files)} files")
                        
                    except Exception as batch_error:
                        error_msg = f"Batch {batch_num} failed: {batch_error}"
                        self.logger.error(error_msg)
                        errors.append(error_msg)
                        files_failed += len(batch)
                        
                        # For critical errors, abort the entire operation
                        if len(errors) > 5:  # Too many batch failures
                            raise Exception("Too many batch failures, aborting cleanup")
                
                self.logger.info(f"Cleanup transaction {transaction_id} completed successfully")
                
        except Exception as e:
            error_msg = f"Cleanup execution failed: {e}"
            self.logger.error(error_msg)
            errors.append(error_msg)
            rollback_performed = True
            
            # The transaction context manager will handle rollback automatically
        
        # Create result
        end_time = time.time()
        result = CleanupResult(
            plan_id=plan.plan_id,
            started_at=start_time,
            completed_at=end_time,
            success=len(errors) == 0,
            files_deleted=files_deleted,
            files_failed_to_delete=files_failed,
            size_freed_mb=size_freed,
            errors=errors,
            warnings=warnings,
            transaction_id=transaction_id,
            rollback_performed=rollback_performed
        )
        
        # Update statistics
        self._cleanup_history.append(result)
        if result.success:
            self._last_cleanup_time = end_time
            self._total_files_deleted += files_deleted
            self._total_space_freed_mb += size_freed
        
        # Log result summary
        if result.success:
            self.logger.info(f"Cleanup {plan.plan_id} completed successfully: "
                           f"deleted {files_deleted} files ({size_freed:.1f}MB) "
                           f"in {result.duration_seconds:.1f}s")
        else:
            self.logger.error(f"Cleanup {plan.plan_id} failed: "
                            f"deleted {files_deleted}/{len(plan.files_to_delete)} files, "
                            f"{len(errors)} errors")
        
        return result
    
    def run_full_cleanup(self, dry_run: bool = False) -> Tuple[CleanupPlan, Optional[CleanupResult]]:
        """
        Run a complete cleanup cycle: plan creation and execution
        
        Args:
            dry_run: If True, only create plan without executing
            
        Returns:
            Tuple of (CleanupPlan, CleanupResult or None if dry_run)
        """
        self.logger.info(f"Starting full cleanup cycle (dry_run={dry_run})")
        
        try:
            # Create cleanup plan
            plan = self.create_cleanup_plan(dry_run=dry_run)
            
            if dry_run:
                self.logger.info("Dry run completed - no files were deleted")
                return plan, None
            
            # Execute plan if not dry run
            if len(plan.files_to_delete) == 0:
                self.logger.info("No files to delete - cleanup not needed")
                return plan, None
            
            result = self.execute_cleanup_plan(plan)
            return plan, result
            
        except Exception as e:
            self.logger.error(f"Full cleanup cycle failed: {e}")
            raise
    
    def cleanup_orphaned_files(self) -> int:
        """
        Quick cleanup of orphaned temporary files
        
        Returns:
            Number of orphaned files cleaned up
        """
        self.logger.info("Running orphaned file cleanup")
        
        try:
            return self.file_manager.cleanup_orphaned_temp_files()
        except Exception as e:
            self.logger.error(f"Orphaned file cleanup failed: {e}")
            raise
    
    def _analyze_cleanup_safety(self, files_to_delete: List[FileInfo], 
                               files_to_keep: List[FileInfo]) -> List[str]:
        """
        Analyze the safety of a proposed cleanup operation
        
        Args:
            files_to_delete: Files that would be deleted
            files_to_keep: Files that would be kept
            
        Returns:
            List of safety warnings
        """
        warnings = []
        
        # Check deletion percentage
        total_files = len(files_to_delete) + len(files_to_keep)
        if total_files > 0:
            deletion_percentage = (len(files_to_delete) / total_files) * 100
            if deletion_percentage > 70:
                warnings.append(f"HIGH DELETION RATE: {deletion_percentage:.1f}% of files will be deleted")
            elif deletion_percentage > 50:
                warnings.append(f"MODERATE DELETION RATE: {deletion_percentage:.1f}% of files will be deleted")
        
        # Check if we're deleting critical files
        critical_deletions = [f for f in files_to_delete if f.priority == FilePriority.CRITICAL]
        if critical_deletions:
            warnings.append(f"CRITICAL FILES: {len(critical_deletions)} critical files will be deleted")
        
        # Check if we're keeping any recent files
        recent_kept = [f for f in files_to_keep if f.age_hours < 24]
        if len(recent_kept) == 0:
            warnings.append("NO RECENT FILES: No files from the last 24 hours will be kept")
        
        # Check for very large cleanup operations
        total_size_mb = sum(f.size_bytes for f in files_to_delete) / (1024 * 1024)
        if total_size_mb > 500:
            warnings.append(f"LARGE CLEANUP: {total_size_mb:.1f}MB will be deleted")
        
        # Check if we're deleting transaction files
        transaction_deletions = [f for f in files_to_delete if f.is_transaction]
        if transaction_deletions:
            warnings.append(f"TRANSACTION FILES: {len(transaction_deletions)} transaction files will be deleted")
        
        return warnings
    
    def _apply_safety_limits(self, files_to_delete: List[FileInfo]) -> List[FileInfo]:
        """
        Apply safety limits to the files scheduled for deletion
        
        Args:
            files_to_delete: Original list of files to delete
            
        Returns:
            Limited list of files to delete
        """
        if len(files_to_delete) <= self.max_files_per_cleanup:
            total_size_mb = sum(f.size_bytes for f in files_to_delete) / (1024 * 1024)
            if total_size_mb <= self.max_size_per_cleanup_mb:
                return files_to_delete
        
        # Sort by priority (delete low priority first) and age (delete old first)
        def sort_key(f: FileInfo) -> Tuple[int, float, float]:
            priority_order = {
                FilePriority.LOW: 0,
                FilePriority.MEDIUM: 1,
                FilePriority.HIGH: 2,
                FilePriority.CRITICAL: 3
            }
            return (priority_order[f.priority], -f.age_hours, f.size_bytes)
        
        sorted_files = sorted(files_to_delete, key=sort_key)
        
        # Apply limits
        limited_files = []
        total_size = 0
        
        for file_info in sorted_files:
            if len(limited_files) >= self.max_files_per_cleanup:
                break
            if (total_size + file_info.size_bytes) / (1024 * 1024) > self.max_size_per_cleanup_mb:
                break
            
            limited_files.append(file_info)
            total_size += file_info.size_bytes
        
        if len(limited_files) < len(files_to_delete):
            removed_count = len(files_to_delete) - len(limited_files)
            self.logger.warning(f"Applied safety limits: removed {removed_count} files from deletion plan")
        
        return limited_files
    
    def _estimate_cleanup_time(self, files_to_delete: List[FileInfo]) -> float:
        """
        Estimate the time required to clean up the specified files
        
        Args:
            files_to_delete: Files that will be deleted
            
        Returns:
            Estimated time in seconds
        """
        if not files_to_delete:
            return 0.0
        
        # Base time estimates (conservative)
        base_time_per_file = 0.1  # 100ms per file for I/O operations
        backup_time_per_mb = 0.5   # 0.5 seconds per MB for backup creation
        
        file_count = len(files_to_delete)
        total_size_mb = sum(f.size_bytes for f in files_to_delete) / (1024 * 1024)
        
        estimated_time = (file_count * base_time_per_file) + (total_size_mb * backup_time_per_mb)
        
        # Add overhead for transaction management
        estimated_time *= 1.2
        
        return estimated_time
    
    def _summarize_policy_decisions(self, all_files: List[FileInfo], 
                                   retention_decisions: Dict[Path, bool]) -> Dict[str, int]:
        """
        Summarize how many files each policy decided to delete
        
        Args:
            all_files: All files that were evaluated
            retention_decisions: Policy decisions
            
        Returns:
            Dictionary mapping policy names to deleted file counts
        """
        # This is a simplified summary - in practice, we'd need more detailed
        # tracking of which policy made each decision
        summary = {}
        
        for policy in self.retention_policies.policies:
            if policy.enabled:
                summary[policy.name] = 0
        
        # Count files marked for deletion
        deleted_count = sum(1 for keep in retention_decisions.values() if not keep)
        
        # For now, attribute deletions evenly across active policies
        # In a more sophisticated implementation, each policy would track its decisions
        active_policies = [p.name for p in self.retention_policies.policies if p.enabled]
        if active_policies:
            deletions_per_policy = deleted_count // len(active_policies)
            remainder = deleted_count % len(active_policies)
            
            for i, policy_name in enumerate(active_policies):
                summary[policy_name] = deletions_per_policy
                if i < remainder:
                    summary[policy_name] += 1
        
        return summary
    
    def get_cleanup_statistics(self) -> Dict[str, Union[int, float, List]]:
        """
        Get comprehensive cleanup statistics
        
        Returns:
            Dictionary with cleanup statistics
        """
        recent_results = self._cleanup_history[-10:]  # Last 10 cleanups
        
        return {
            'total_cleanups_performed': len(self._cleanup_history),
            'total_files_deleted': self._total_files_deleted,
            'total_space_freed_mb': self._total_space_freed_mb,
            'last_cleanup_time': self._last_cleanup_time,
            'recent_success_rate': (
                sum(1 for r in recent_results if r.success) / len(recent_results) * 100
                if recent_results else 0
            ),
            'average_cleanup_duration': (
                sum(r.duration_seconds for r in recent_results) / len(recent_results)
                if recent_results else 0
            ),
            'retention_policies_active': len([p for p in self.retention_policies.policies if p.enabled]),
            'retention_policies_total': len(self.retention_policies.policies),
            'max_files_per_cleanup': self.max_files_per_cleanup,
            'max_size_per_cleanup_mb': self.max_size_per_cleanup_mb
        }
    
    def get_service_status(self) -> Dict[str, Union[str, int, float, bool]]:
        """
        Get current service status
        
        Returns:
            Dictionary with service status information
        """
        file_manager_status = self.file_manager.get_manager_status()
        
        return {
            'service_initialized': True,
            'file_manager_base_path': file_manager_status.get('base_path', 'unknown'),
            'current_file_count': file_manager_status.get('total_files', 0),
            'current_total_size_mb': file_manager_status.get('total_size_mb', 0),
            'orphaned_files_detected': file_manager_status.get('orphaned_files', 0),
            'retention_policies_configured': len(self.retention_policies.policies),
            'retention_policies_enabled': len([p for p in self.retention_policies.policies if p.enabled]),
            'cleanup_history_entries': len(self._cleanup_history),
            'last_cleanup_success': (
                self._cleanup_history[-1].success 
                if self._cleanup_history else None
            ),
            'time_since_last_cleanup_hours': (
                (time.time() - self._last_cleanup_time) / 3600
                if self._last_cleanup_time else None
            )
        }