#!/usr/bin/env python3
"""Test plotting functionality."""

from pathlib import Path
import sys

# Ensure the project root is on sys.path for imports when running this file directly
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Acasm.engine import Engine

def test_plot_basic():
    """Test basic plotting functionality."""
    eng = Engine()
    
    # Test plotting expression
    result = eng.plot('sin(x), x')
    assert 'Plot of sin(x):' in result
    assert 'X: -10 to 10, Y: -1 to 1' in result
    assert '*' in result  # Should contain plot characters
    
    # Test plotting with custom range
    result = eng.plot('x^2, x, -3, 3')
    assert 'Plot of x**2:' in result
    assert 'X: -3 to 3' in result
    assert '*' in result
    
    print("Basic plotting tests passed!")

def test_plot_function():
    """Test plotting defined functions."""
    eng = Engine()
    
    # Define a function
    fname, lam = eng.define_function('f', ['x'], 'x^2 - 4')
    assert fname == 'f'
    
    # Plot the function
    result = eng.plot('f')
    assert 'Plot of function f(x):' in result
    assert '*' in result
    assert 'X: -10 to 10' in result
    
    # Plot function with custom range
    result = eng.plot('f, x, -5, 5')
    assert 'Plot of function f(x):' in result
    assert 'X: -5 to 5' in result
    
    print("Function plotting tests passed!")

def test_plot_edge_cases():
    """Test edge case scenarios."""
    eng = Engine()
    
    # Test invalid usage - should raise exception
    try:
        eng.plot('')
        assert False, "Should have raised exception for empty input"
    except ValueError as e:
        assert 'Usage:' in str(e)
    
    # Test expression without variable - should raise exception
    try:
        eng.plot('x^2')
        assert False, "Should have raised exception for missing variable"
    except ValueError as e:
        assert 'specify variable' in str(e)
    
    # Test custom dimensions
    result = eng.plot('sin(x), x, -3, 3, 30, 10')
    lines = result.split('\n')
    # Should have title + 10 plot lines + range line
    assert len([line for line in lines if line and not line.startswith('Plot of') and not line.startswith('X:')]) == 10
    
    print("Edge case tests passed!")

if __name__ == "__main__":
    test_plot_basic()
    test_plot_function()
    test_plot_edge_cases()
    print("All plotting tests passed!")