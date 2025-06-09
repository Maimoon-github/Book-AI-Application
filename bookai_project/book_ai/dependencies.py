"""
Dependency Manager for Book AI RAG Application.

This module provides robust dependency checking and versioning to
ensure the application can run with the available dependencies,
falling back gracefully if needed.
"""

import sys
import warnings
import importlib
import importlib.util
from typing import Dict, Any, Optional

class DependencyManager:
    """
    Manager for checking and tracking dependencies.
    """
    def __init__(self):
        """Initialize dependency tracking."""
        self.dependencies = {
            "numpy": {"required": False, "available": False, "version": None, "min_version": "1.19.0"},
            "faiss": {"required": False, "available": False, "version": None, "min_version": None},
            "nltk": {"required": False, "available": False, "version": None, "min_version": "3.6.0"},
            "sentence_transformers": {"required": False, "available": False, "version": None, "min_version": "2.2.0"},
            "rank_bm25": {"required": False, "available": False, "version": None, "min_version": None},
            "torch": {"required": False, "available": False, "version": None, "min_version": "1.7.0"},
            "PyPDF2": {"required": True, "available": False, "version": None, "min_version": "2.0.0"},
            "transformers": {"required": False, "available": False, "version": None, "min_version": "4.0.0"},
        }
        
    def check_all(self):
        """Check all dependencies and report status."""
        for package_name in self.dependencies:
            self._check_package(package_name)
            
        # Print status to stderr for logging
        print("RAG Dependencies Status:", file=sys.stderr)
        for package_name, info in self.dependencies.items():
            status_str = f"✓ Available {info['version']}" if info["available"] else "✗ Missing"
            required_str = "(required)" if info["required"] else ""
            print(f"  - {package_name}: {status_str} {required_str}", file=sys.stderr)
        
        # Check if required dependencies are met
        missing_required = [pkg for pkg, info in self.dependencies.items() 
                           if info["required"] and not info["available"]]
        if missing_required:
            required_list = ", ".join(missing_required)
            warnings.warn(f"Required dependencies missing: {required_list}. Some features will be disabled.")
      def _check_package(self, package_name: str) -> None:
        """Check if a package is available and get its version."""
        try:
            # Special case for faiss-cpu package which is imported as 'faiss'
            if package_name == "faiss":
                try:
                    import faiss
                    if hasattr(faiss, "IndexFlatIP"):
                        self.dependencies[package_name]["available"] = True
                        self.dependencies[package_name]["version"] = getattr(faiss, "__version__", "unknown")
                        return
                except ImportError:
                    # Try faiss-gpu as a fallback
                    try:
                        import faiss
                        if hasattr(faiss, "IndexFlatIP"):
                            self.dependencies[package_name]["available"] = True
                            self.dependencies[package_name]["version"] = getattr(faiss, "__version__", "unknown")
                            return
                    except ImportError:
                        pass
                
                self.dependencies[package_name]["available"] = False
                return
                
            # Normal package import
            module = importlib.import_module(package_name)
            
            # Check for specific attributes to confirm functionality
            if package_name == "PyPDF2" and not hasattr(module, "PdfReader"):
                print(f"PyPDF2 found but missing PdfReader class. Available attributes: {dir(module)[:10]}", file=sys.stderr)
                raise ImportError("PyPDF2 found but missing PdfReader class")
            
            # Get version if available
            version = getattr(module, "__version__", "unknown")
            
            # Update dependency status
            self.dependencies[package_name]["available"] = True
            self.dependencies[package_name]["version"] = version
        
        except (ImportError, ModuleNotFoundError) as e:
            # Log the error for debugging
            print(f"Failed to import {package_name}: {str(e)}", file=sys.stderr)
            
            # Package not available
            self.dependencies[package_name]["available"] = False
    
    def is_available(self, package_name: str) -> bool:
        """Check if a package is available."""
        return self.dependencies.get(package_name, {}).get("available", False)
    
    def get_version(self, package_name: str) -> Optional[str]:
        """Get the version of a package if available."""
        return self.dependencies.get(package_name, {}).get("version", None)


# Singleton instance for global use
dependency_manager = DependencyManager()
