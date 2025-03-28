#!/usr/bin/env python
"""
Run test coverage analysis for the utils package.
"""
import sys
import os
import coverage

# Add the project root to the path so that utils can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

if __name__ == "__main__":
    # Create a coverage object
    cov = coverage.Coverage(
        source=["utils"],
        omit=["utils/tests/*"],
        branch=True,
    )
    
    # Start measuring coverage
    cov.start()
    
    # Import and run all tests
    import unittest
    test_suite = unittest.defaultTestLoader.discover('.', pattern='test_*.py')
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    # Stop measuring coverage
    cov.stop()
    
    # Print the result summary
    print("\nTest Summary:")
    print(f"Ran {result.testsRun} tests")
    if result.wasSuccessful():
        print("All tests passed!")
    else:
        print(f"Tests failed: {len(result.failures)} failures, {len(result.errors)} errors")
    
    # Print coverage report to terminal
    print("\nCoverage Summary:")
    cov.report()
    
    # Generate HTML report
    html_dir = os.path.join(os.path.dirname(__file__), 'coverage_html')
    print(f"\nGenerating HTML report in {html_dir}")
    cov.html_report(directory=html_dir)
    
    # Exit with non-zero code if tests failed
    sys.exit(not result.wasSuccessful()) 