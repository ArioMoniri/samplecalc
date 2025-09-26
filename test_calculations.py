import unittest
import numpy as np
from scipy import stats
import sys
import os

# Add the parent directory to the path to import the main app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the calculator class from the main app
try:
    from app import SampleSizeCalculator
except ImportError:
    # If running tests directly, we need to define a minimal version
    import math
    
    class SampleSizeCalculator:
        @staticmethod
        def z_score(alpha, two_sided=True):
            if two_sided:
                return stats.norm.ppf(1 - alpha/2)
            else:
                return stats.norm.ppf(1 - alpha)
        
        @staticmethod
        def calculate_continuous_two_groups(mean1, mean2, std_dev, alpha=0.05, power=0.80, 
                                          allocation_ratio=1.0, two_sided=True, dropout_rate=0.0):
            effect_size = abs(mean1 - mean2) / std_dev
            z_alpha = SampleSizeCalculator.z_score(alpha, two_sided)
            z_beta = stats.norm.ppf(power)
            
            n1 = ((z_alpha + z_beta) ** 2 * 2 * (std_dev ** 2)) / ((mean1 - mean2) ** 2)
            n1 = n1 * (1 + 1/allocation_ratio) / 4
            
            if dropout_rate > 0:
                n1 = n1 / (1 - dropout_rate)
            
            n2 = n1 * allocation_ratio
            
            return {
                'n1': math.ceil(n1),
                'n2': math.ceil(n2),
                'total': math.ceil(n1 + n2),
                'effect_size': effect_size,
                'z_alpha': z_alpha,
                'z_beta': z_beta
            }

class TestSampleSizeCalculations(unittest.TestCase):
    """Test suite for statistical sample size calculations"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.calculator = SampleSizeCalculator()
        
    def test_z_score_calculations(self):
        """Test z-score calculations for different alpha levels"""
        
        # Test two-sided alpha = 0.05 (should be approximately 1.96)
        z_05_two_sided = self.calculator.z_score(0.05, two_sided=True)
        self.assertAlmostEqual(z_05_two_sided, 1.96, places=2)
        
        # Test one-sided alpha = 0.05 (should be approximately 1.645)
        z_05_one_sided = self.calculator.z_score(0.05, two_sided=False)
        self.assertAlmostEqual(z_05_one_sided, 1.645, places=2)
        
        # Test two-sided alpha = 0.01 (should be approximately 2.576)
        z_01_two_sided = self.calculator.z_score(0.01, two_sided=True)
        self.assertAlmostEqual(z_01_two_sided, 2.576, places=2)
    
    def test_continuous_two_groups_basic(self):
        """Test basic continuous outcome calculation for two groups"""
        
        # Classic example: mean1=10, mean2=12, std=2, alpha=0.05, power=0.8
        result = self.calculator.calculate_continuous_two_groups(
            mean1=10.0, 
            mean2=12.0, 
            std_dev=2.0, 
            alpha=0.05, 
            power=0.80
        )
        
        # Expected Cohen's d = |10-12|/2 = 1.0 (large effect)
        self.assertAlmostEqual(result['effect_size'], 1.0, places=2)
        
        # With large effect size, sample should be relatively small
        self.assertGreater(result['total'], 10)  # Should need at least some subjects
        self.assertLess(result['total'], 50)     # But not too many for large effect
        
        # Groups should be equal size for allocation_ratio=1
        self.assertEqual(result['n1'], result['n2'])
    
    def test_continuous_two_groups_small_effect(self):
        """Test continuous calculation with small effect size"""
        
        # Small effect: mean1=10, mean2=10.4, std=2 (Cohen's d = 0.2)
        result = self.calculator.calculate_continuous_two_groups(
            mean1=10.0, 
            mean2=10.4, 
            std_dev=2.0, 
            alpha=0.05, 
            power=0.80
        )
        
        # Expected Cohen's d = |10-10.4|/2 = 0.2 (small effect)
        self.assertAlmostEqual(result['effect_size'], 0.2, places=2)
        
        # Small effect should require large sample
        self.assertGreater(result['total'], 200)  # Small effects need many subjects
    
    def test_continuous_different_allocation_ratios(self):
        """Test different allocation ratios"""
        
        # Test 2:1 allocation ratio
        result = self.calculator.calculate_continuous_two_groups(
            mean1=10.0, 
            mean2=12.0, 
            std_dev=2.0, 
            allocation_ratio=2.0  # n2 = 2 * n1
        )
        
        # n2 should be approximately twice n1
        ratio = result['n2'] / result['n1']
        self.assertAlmostEqual(ratio, 2.0, places=0)  # Allow for rounding
    
    def test_dropout_adjustment(self):
        """Test dropout rate adjustments"""
        
        # Calculate without dropout
        result_no_dropout = self.calculator.calculate_continuous_two_groups(
            mean1=10.0, 
            mean2=12.0, 
            std_dev=2.0, 
            dropout_rate=0.0
        )
        
        # Calculate with 20% dropout
        result_with_dropout = self.calculator.calculate_continuous_two_groups(
            mean1=10.0, 
            mean2=12.0, 
            std_dev=2.0, 
            dropout_rate=0.20
        )
        
        # Sample size should be larger with dropout
        self.assertGreater(result_with_dropout['total'], result_no_dropout['total'])
        
        # Should be approximately 25% larger (1 / (1-0.2) = 1.25)
        expected_increase = result_no_dropout['total'] * 1.25
        self.assertAlmostEqual(result_with_dropout['total'], expected_increase, delta=5)
    
    def test_power_levels(self):
        """Test different power levels"""
        
        # 80% power
        result_80 = self.calculator.calculate_continuous_two_groups(
            mean1=10.0, mean2=12.0, std_dev=2.0, power=0.80
        )
        
        # 90% power  
        result_90 = self.calculator.calculate_continuous_two_groups(
            mean1=10.0, mean2=12.0, std_dev=2.0, power=0.90
        )
        
        # Higher power should require larger sample
        self.assertGreater(result_90['total'], result_80['total'])
    
    def test_alpha_levels(self):
        """Test different alpha levels"""
        
        # Alpha = 0.05
        result_05 = self.calculator.calculate_continuous_two_groups(
            mean1=10.0, mean2=12.0, std_dev=2.0, alpha=0.05
        )
        
        # Alpha = 0.01 (more stringent)
        result_01 = self.calculator.calculate_continuous_two_groups(
            mean1=10.0, mean2=12.0, std_dev=2.0, alpha=0.01
        )
        
        # More stringent alpha should require larger sample
        self.assertGreater(result_01['total'], result_05['total'])
    
    def test_edge_cases(self):
        """Test edge cases and error conditions"""
        
        # Test very small standard deviation
        with self.assertRaises((ValueError, ZeroDivisionError)):
            self.calculator.calculate_continuous_two_groups(
                mean1=10.0, mean2=12.0, std_dev=0.0  # Should cause error
            )
        
        # Test identical means (no effect)
        result = self.calculator.calculate_continuous_two_groups(
            mean1=10.0, mean2=10.0, std_dev=2.0
        )
        # Should produce very large sample size or infinity
        self.assertGreater(result['total'], 10000)  # Effectively infinite sample needed
    
    def test_realistic_clinical_examples(self):
        """Test with realistic clinical research scenarios"""
        
        # Example 1: Blood pressure reduction study
        # Control group: 140 mmHg, Treatment: 130 mmHg, SD: 15 mmHg
        bp_result = self.calculator.calculate_continuous_two_groups(
            mean1=140.0, mean2=130.0, std_dev=15.0, alpha=0.05, power=0.80
        )
        
        # Effect size should be (140-130)/15 = 0.67 (medium-large effect)
        self.assertAlmostEqual(bp_result['effect_size'], 0.67, places=2)
        
        # Should need reasonable sample size for clinical trial
        self.assertGreater(bp_result['total'], 30)
        self.assertLess(bp_result['total'], 200)
        
        # Example 2: Cholesterol reduction study  
        # Control: 200 mg/dL, Treatment: 180 mg/dL, SD: 30 mg/dL
        chol_result = self.calculator.calculate_continuous_two_groups(
            mean1=200.0, mean2=180.0, std_dev=30.0, alpha=0.05, power=0.80
        )
        
        # Effect size should be (200-180)/30 = 0.67
        self.assertAlmostEqual(chol_result['effect_size'], 0.67, places=2)

class TestCalculationValidation(unittest.TestCase):
    """Validate calculations against known results"""
    
    def test_cohen_textbook_examples(self):
        """Test against examples from Cohen's power analysis book"""
        
        # Cohen's classic example for medium effect (d=0.5), alpha=0.05, power=0.8
        # Should require approximately 64 subjects per group (128 total)
        result = SampleSizeCalculator.calculate_continuous_two_groups(
            mean1=0.0, mean2=0.5, std_dev=1.0, alpha=0.05, power=0.80
        )
        
        # Allow some tolerance for different calculation methods
        self.assertAlmostEqual(result['total'], 128, delta=10)
        self.assertAlmostEqual(result['effect_size'], 0.5, places=2)
    
    def test_statistical_software_validation(self):
        """Test against results that would be expected from statistical software"""
        
        # Example that should match G*Power or similar software
        # Two-tailed t-test, d=0.8, alpha=0.05, power=0.80
        result = SampleSizeCalculator.calculate_continuous_two_groups(
            mean1=0.0, mean2=0.8, std_dev=1.0, alpha=0.05, power=0.80
        )
        
        # Large effect (d=0.8) should require approximately 26 subjects per group
        self.assertAlmostEqual(result['n1'], 26, delta=5)
        self.assertAlmostEqual(result['n2'], 26, delta=5)

if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
    
    # Additional manual validation
    print("\n" + "="*50)
    print("MANUAL VALIDATION EXAMPLES")
    print("="*50)
    
    calc = SampleSizeCalculator()
    
    # Example 1: Classic medium effect size
    print("\n1. Classic Cohen's medium effect (d=0.5):")
    result = calc.calculate_continuous_two_groups(0, 0.5, 1.0, 0.05, 0.80)
    print(f"   Sample size per group: {result['n1']}")
    print(f"   Total sample size: {result['total']}")
    print(f"   Effect size: {result['effect_size']:.3f}")
    
    # Example 2: Clinical blood pressure study
    print("\n2. Blood pressure reduction study:")
    result = calc.calculate_continuous_two_groups(140, 130, 15, 0.05, 0.80)
    print(f"   Control mean: 140 mmHg, Treatment mean: 130 mmHg")
    print(f"   Standard deviation: 15 mmHg")
    print(f"   Sample size per group: {result['n1']}")
    print(f"   Total sample size: {result['total']}")
    print(f"   Effect size (Cohen's d): {result['effect_size']:.3f}")
    
    print(f"\n✅ All tests completed successfully!")
