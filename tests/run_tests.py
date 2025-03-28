#!/usr/bin/env python
"""
Run all tests for the utils package.
"""
import unittest
import sys
import os

# Add the parent directory to the path so that utils can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

if __name__ == "__main__":
    # Discover and run all tests in the current directory
    test_suite = unittest.defaultTestLoader.discover('.', pattern='test_*.py')
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    # Print a summary
    print(f"\nTest Summary:")
    print(f"Ran {result.testsRun} tests")
    if result.wasSuccessful():
        print("All tests passed!")
    else:
        print(f"Tests failed: {len(result.failures)} failures, {len(result.errors)} errors")
    
    # Exit with non-zero code if tests failed
    sys.exit(not result.wasSuccessful()) 