import numpy as np
import pytest

from ps3.preprocessing import Winsorizer

# TODO: Test your implementation of a simple Winsorizer
@pytest.mark.parametrize(
    "lower_quantile, upper_quantile", [(0, 1), (0.05, 0.95), (0.5, 0.5)]
)
def test_winsorizer(lower_quantile, upper_quantile):

    X = np.random.randint(0,200, size = 30)
    #X = np.random.normal(0, 1, 1000)

        # Reshape to 2D (your Winsorizer expects 2D arrays)
    #X = X.reshape(-1, 1)  # Makes it (1000, 1) instead of (1000,)
    
    # Create and fit the winsorizer
    winsorizer = Winsorizer(lower_quantile, upper_quantile)
    winsorizer.fit(X)
    
    # Transform the data
    X_transformed = winsorizer.transform(X)
    
    # Test 1: Check that fit() created the required attributes
    assert hasattr(winsorizer, 'lower_quantile_'), "fit() should create lower_quantile_"
    assert hasattr(winsorizer, 'upper_quantile_'), "fit() should create upper_quantile_"
    
    # Test 2: Check that shape is preserved
    assert X_transformed.shape == X.shape, "Transform should preserve shape"
    
    # Test 3: Check that all values are within the quantile bounds
    assert np.all(X_transformed >= winsorizer.lower_quantile_), \
        "All values should be >= lower quantile"
    assert np.all(X_transformed <= winsorizer.upper_quantile_), \
        "All values should be <= upper quantile"
    
    # Test 4: Check that the computed quantiles are correct
    expected_lower = np.quantile(X, lower_quantile, axis=0)
    expected_upper = np.quantile(X, upper_quantile, axis=0)
    assert np.allclose(winsorizer.lower_quantile_, expected_lower), \
        "Lower quantile doesn't match expected value"
    assert np.allclose(winsorizer.upper_quantile_, expected_upper), \
        "Upper quantile doesn't match expected value"
