"""
Retention Policies - Define how long to keep different types of state files

This module implements intelligent retention strategies to balance storage efficiency
with data availability and rollback capabilities.

Key Features:
- Time-based retention (keep files for specific durations)
- Count-based retention (keep N most recent files)
- Size-based retention (keep files within storage limits)
- Hierarchical retention (24h recent + 7d hourly + weekly)
- File categorization and prioritization
- Safe deletion with validation
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
import time

logger = logging.getLogger(__name__)


class RetentionStrategy(Enum):
    """Available retention strategies"""
    TIME_BASED = "time_based"
    COUNT_BASED = "count_based" 
    SIZE_BASED = "size_based"
    HIERARCHICAL = "hierarchical"


class FilePriority(Enum):
    """File priority levels for retention decisions"""
    CRITICAL = "critical"      # Never delete (recent backups, transactions)
    HIGH = "high"             # Keep longer (hourly snapshots)  
    MEDIUM = "medium"         # Standard retention (regular saves)
    LOW = "low"              # Delete first (temp files, old snapshots)


@dataclass
class FileInfo:
    """Metadata about a state file"""
    path: Path
    size_bytes: int
    created_time: float
    modified_time: float
    priority: FilePriority
    is_temp: bool = False
    is_transaction: bool = False
    is_orphaned: bool = False
    
    @property
    def age_hours(self) -> float:
        """Get file age in hours"""
        return (time.time() - self.modified_time) / 3600
    
    @property
    def age_days(self) -> float:
        """Get file age in days"""
        return self.age_hours / 24
    
    def __str__(self) -> str:
        return f"FileInfo({self.path.name}, {self.size_bytes}B, {self.age_hours:.1f}h old, {self.priority.value})"


class RetentionPolicy(ABC):
    """Base class for retention policies"""
    
    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.enabled = enabled
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def should_retain(self, file_info: FileInfo) -> bool:
        """
        Determine if a file should be retained
        
        Args:
            file_info: File metadata
            
        Returns:
            True if file should be kept, False if it can be deleted
        """
        pass
    
    @abstractmethod
    def get_policy_description(self) -> str:
        """Get human-readable policy description"""
        pass
    
    def __str__(self) -> str:
        return f"{self.name}: {self.get_policy_description()}"


class TimeBasedRetentionPolicy(RetentionPolicy):
    """Retain files based on age thresholds"""
    
    def __init__(self, name: str, max_age_hours: float, enabled: bool = True):
        super().__init__(name, enabled)
        self.max_age_hours = max_age_hours
    
    def should_retain(self, file_info: FileInfo) -> bool:
        if not self.enabled:
            return True
            
        # Always keep critical files
        if file_info.priority == FilePriority.CRITICAL:
            return True
            
        # Check age threshold
        should_keep = file_info.age_hours <= self.max_age_hours
        
        if not should_keep:
            self.logger.debug(f"File {file_info.path.name} exceeds max age "
                            f"({file_info.age_hours:.1f}h > {self.max_age_hours}h)")
        
        return should_keep
    
    def get_policy_description(self) -> str:
        return f"Keep files newer than {self.max_age_hours} hours"


class CountBasedRetentionPolicy(RetentionPolicy):
    """Retain N most recent files per priority level"""
    
    def __init__(self, name: str, max_count_per_priority: Dict[FilePriority, int], enabled: bool = True):
        super().__init__(name, enabled)
        self.max_count_per_priority = max_count_per_priority
    
    def should_retain(self, file_info: FileInfo) -> bool:
        # This policy requires the full file list to make decisions
        # It's implemented in should_retain_batch method
        return True
    
    def should_retain_batch(self, files: List[FileInfo]) -> Dict[Path, bool]:
        """
        Batch processing for count-based retention
        
        Args:
            files: List of all files to evaluate
            
        Returns:
            Dictionary mapping file paths to retention decisions
        """
        if not self.enabled:
            return {f.path: True for f in files}
        
        retention_map = {}
        
        # Group files by priority
        priority_groups = {}
        for file_info in files:
            priority = file_info.priority
            if priority not in priority_groups:
                priority_groups[priority] = []
            priority_groups[priority].append(file_info)
        
        # Apply count limits per priority
        for priority, file_list in priority_groups.items():
            max_count = self.max_count_per_priority.get(priority, float('inf'))
            
            # Sort by modification time (newest first)
            sorted_files = sorted(file_list, key=lambda f: f.modified_time, reverse=True)
            
            # Keep the N most recent files
            for i, file_info in enumerate(sorted_files):
                should_keep = i < max_count
                retention_map[file_info.path] = should_keep
                
                if not should_keep:
                    self.logger.debug(f"File {file_info.path.name} exceeds count limit "
                                    f"for {priority.value} priority (position {i+1} > {max_count})")
        
        return retention_map
    
    def get_policy_description(self) -> str:
        limits = [f"{p.value}: {c}" for p, c in self.max_count_per_priority.items()]
        return f"Keep max files per priority: {', '.join(limits)}"


class SizeBasedRetentionPolicy(RetentionPolicy):
    """Retain files within total size limits"""
    
    def __init__(self, name: str, max_total_size_mb: float, enabled: bool = True):
        super().__init__(name, enabled)
        self.max_total_size_bytes = max_total_size_mb * 1024 * 1024
    
    def should_retain(self, file_info: FileInfo) -> bool:
        # This policy requires the full file list to make decisions
        # It's implemented in should_retain_batch method
        return True
    
    def should_retain_batch(self, files: List[FileInfo]) -> Dict[Path, bool]:
        """
        Batch processing for size-based retention
        
        Args:
            files: List of all files to evaluate
            
        Returns:
            Dictionary mapping file paths to retention decisions
        """
        if not self.enabled:
            return {f.path: True for f in files}
        
        # Sort files by priority (critical first) then by age (newest first)
        def sort_key(f: FileInfo) -> Tuple[int, float]:
            priority_order = {
                FilePriority.CRITICAL: 0,
                FilePriority.HIGH: 1, 
                FilePriority.MEDIUM: 2,
                FilePriority.LOW: 3
            }
            return (priority_order[f.priority], -f.modified_time)
        
        sorted_files = sorted(files, key=sort_key)
        
        retention_map = {}
        cumulative_size = 0
        
        for file_info in sorted_files:
            # Always keep critical files regardless of size
            if file_info.priority == FilePriority.CRITICAL:
                retention_map[file_info.path] = True
                cumulative_size += file_info.size_bytes
                continue
            
            # Check if adding this file would exceed size limit
            would_exceed = (cumulative_size + file_info.size_bytes) > self.max_total_size_bytes
            
            if would_exceed:
                retention_map[file_info.path] = False
                self.logger.debug(f"File {file_info.path.name} would exceed size limit "
                                f"({cumulative_size + file_info.size_bytes} > {self.max_total_size_bytes})")
            else:
                retention_map[file_info.path] = True
                cumulative_size += file_info.size_bytes
        
        return retention_map
    
    def get_policy_description(self) -> str:
        size_mb = self.max_total_size_bytes / (1024 * 1024)
        return f"Keep files within {size_mb:.1f}MB total size limit"


class HierarchicalRetentionPolicy(RetentionPolicy):
    """
    Hierarchical retention with multiple time periods:
    - Recent: Keep all files (24 hours)  
    - Hourly: Keep 1 file per hour (7 days)
    - Daily: Keep 1 file per day (30 days)
    - Weekly: Keep 1 file per week (indefinitely)
    """
    
    def __init__(self, 
                 name: str,
                 recent_hours: float = 24,
                 hourly_days: int = 7,
                 daily_days: int = 30,
                 enabled: bool = True):
        super().__init__(name, enabled)
        self.recent_hours = recent_hours
        self.hourly_days = hourly_days  
        self.daily_days = daily_days
    
    def should_retain(self, file_info: FileInfo) -> bool:
        # This policy requires the full file list to make decisions
        # It's implemented in should_retain_batch method
        return True
    
    def should_retain_batch(self, files: List[FileInfo]) -> Dict[Path, bool]:
        """
        Batch processing for hierarchical retention
        
        Args:
            files: List of all files to evaluate
            
        Returns:
            Dictionary mapping file paths to retention decisions  
        """
        if not self.enabled:
            return {f.path: True for f in files}
        
        retention_map = {}
        current_time = time.time()
        
        # Sort files by modification time (newest first)
        sorted_files = sorted(files, key=lambda f: f.modified_time, reverse=True)
        
        # Track what time periods we've covered
        covered_hours: Set[int] = set()
        covered_days: Set[int] = set()  
        covered_weeks: Set[int] = set()
        
        for file_info in sorted_files:
            should_keep = False
            age_hours = file_info.age_hours
            
            # Always keep critical files
            if file_info.priority == FilePriority.CRITICAL:
                should_keep = True
                retention_map[file_info.path] = True
                continue
            
            # Recent period: keep all files
            if age_hours <= self.recent_hours:
                should_keep = True
                self.logger.debug(f"Keeping {file_info.path.name} - recent ({age_hours:.1f}h)")
            
            # Hourly period: keep one file per hour
            elif age_hours <= (self.hourly_days * 24):
                hour_slot = int(age_hours)
                if hour_slot not in covered_hours:
                    covered_hours.add(hour_slot)
                    should_keep = True
                    self.logger.debug(f"Keeping {file_info.path.name} - hourly slot {hour_slot}")
            
            # Daily period: keep one file per day
            elif age_hours <= (self.daily_days * 24):
                day_slot = int(age_hours / 24)
                if day_slot not in covered_days:
                    covered_days.add(day_slot)
                    should_keep = True
                    self.logger.debug(f"Keeping {file_info.path.name} - daily slot {day_slot}")
            
            # Weekly period: keep one file per week (indefinitely)
            else:
                week_slot = int(age_hours / (24 * 7))
                if week_slot not in covered_weeks:
                    covered_weeks.add(week_slot)
                    should_keep = True
                    self.logger.debug(f"Keeping {file_info.path.name} - weekly slot {week_slot}")
            
            retention_map[file_info.path] = should_keep
            
            if not should_keep:
                self.logger.debug(f"Marking {file_info.path.name} for deletion - "
                                f"age {age_hours:.1f}h, no available slots")
        
        return retention_map
    
    def get_policy_description(self) -> str:
        return (f"Hierarchical: {self.recent_hours}h recent, "
                f"{self.hourly_days}d hourly, {self.daily_days}d daily, weekly archive")


class RetentionPolicySet:
    """
    Manages multiple retention policies and combines their decisions
    
    Policies are applied in order, and a file is retained if ANY policy says to keep it.
    This allows for flexible retention strategies that combine different approaches.
    """
    
    def __init__(self, name: str = "default"):
        self.name = name
        self.policies: List[RetentionPolicy] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def add_policy(self, policy: RetentionPolicy):
        """Add a retention policy"""
        self.policies.append(policy)
        self.logger.info(f"Added policy: {policy}")
    
    def remove_policy(self, policy_name: str):
        """Remove a policy by name"""
        self.policies = [p for p in self.policies if p.name != policy_name]
        self.logger.info(f"Removed policy: {policy_name}")
    
    def evaluate_retention(self, files: List[FileInfo]) -> Dict[Path, bool]:
        """
        Evaluate all policies and return retention decisions
        
        Args:
            files: List of files to evaluate
            
        Returns:
            Dictionary mapping file paths to retention decisions
        """
        if not files:
            return {}
        
        self.logger.info(f"Evaluating retention for {len(files)} files using {len(self.policies)} policies")
        
        # Initialize all files as candidates for deletion
        retention_map = {f.path: False for f in files}
        
        # Apply each policy
        for policy in self.policies:
            if not policy.enabled:
                continue
                
            self.logger.debug(f"Applying policy: {policy.name}")
            
            # Check if policy supports batch processing
            if hasattr(policy, 'should_retain_batch'):
                policy_decisions = policy.should_retain_batch(files)
                
                # Update retention map (OR logic - keep if any policy says keep)
                for path, should_keep in policy_decisions.items():
                    if should_keep:
                        retention_map[path] = True
            else:
                # Apply policy individually to each file
                for file_info in files:
                    if policy.should_retain(file_info):
                        retention_map[file_info.path] = True
        
        # Log retention summary
        keep_count = sum(1 for keep in retention_map.values() if keep)
        delete_count = len(files) - keep_count
        
        self.logger.info(f"Retention decision: keep {keep_count}, delete {delete_count}")
        
        # Log files marked for deletion
        for file_info in files:
            if not retention_map[file_info.path]:
                self.logger.debug(f"Marked for deletion: {file_info}")
        
        return retention_map
    
    def get_policies_description(self) -> List[str]:
        """Get descriptions of all active policies"""
        return [str(p) for p in self.policies if p.enabled]
    
    def __str__(self) -> str:
        enabled_count = sum(1 for p in self.policies if p.enabled)
        return f"RetentionPolicySet({self.name}, {enabled_count}/{len(self.policies)} policies active)"


def create_default_retention_policies() -> RetentionPolicySet:
    """
    Create a default set of retention policies suitable for most use cases
    
    Returns:
        Configured RetentionPolicySet with sensible defaults
    """
    policy_set = RetentionPolicySet("default_policies")
    
    # Hierarchical policy for intelligent time-based retention
    hierarchical = HierarchicalRetentionPolicy(
        name="hierarchical_retention",
        recent_hours=24,      # Keep all files from last 24 hours
        hourly_days=7,        # Keep hourly files for 7 days  
        daily_days=30,        # Keep daily files for 30 days
        enabled=True
    )
    policy_set.add_policy(hierarchical)
    
    # Count-based policy to prevent excessive accumulation
    count_limits = {
        FilePriority.CRITICAL: 100,  # Keep up to 100 critical files
        FilePriority.HIGH: 50,       # Keep up to 50 high priority files
        FilePriority.MEDIUM: 30,     # Keep up to 30 medium priority files  
        FilePriority.LOW: 10         # Keep up to 10 low priority files
    }
    count_based = CountBasedRetentionPolicy(
        name="count_based_limits",
        max_count_per_priority=count_limits,
        enabled=True
    )
    policy_set.add_policy(count_based)
    
    # Size-based policy to prevent storage bloat
    size_based = SizeBasedRetentionPolicy(
        name="size_limit",
        max_total_size_mb=500,  # Limit total state files to 500MB
        enabled=True
    )
    policy_set.add_policy(size_based)
    
    return policy_set