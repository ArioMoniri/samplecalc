import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import math
from datetime import datetime

# Configure page
st.set_page_config(
    page_title="ClinCalc Sample Size Calculator",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional medical styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #2E86AB 0%, #A23B72 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .calc-container {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 15px;
        border: 1px solid #dee2e6;
        margin-bottom: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .formula-container {
        background: linear-gradient(145deg, #ffffff, #f8f9fa);
        padding: 2rem;
        border-radius: 15px;
        border: 2px solid #007bff;
        margin: 1.5rem 0;
        box-shadow: 0 4px 20px rgba(0,123,255,0.15);
    }
    
    .result-container {
        background: linear-gradient(145deg, #d4edda, #c3e6cb);
        padding: 2rem;
        border-radius: 15px;
        border: 1px solid #28a745;
        margin: 1.5rem 0;
        box-shadow: 0 4px 15px rgba(40,167,69,0.2);
    }
    
    .results-header {
        background: #6c757d;
        color: white;
        padding: 1rem;
        border-radius: 10px 10px 0 0;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 0;
    }
    
    .results-title {
        background: #ffffff;
        padding: 1.5rem;
        border-bottom: 2px solid #dee2e6;
        font-size: 1.8rem;
        color: #2c3e50;
        font-weight: bold;
        text-align: center;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        border: 2px solid #28a745;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 0.5rem;
    }
    
    .metric-value {
        font-size: 3rem;
        font-weight: bold;
        color: #28a745;
    }
    
    .metric-label {
        font-size: 1.2rem;
        color: #6c757d;
        margin-top: 0.5rem;
        font-weight: bold;
    }
    
    .visualization-container {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .education-container {
        background: #e3f2fd;
        padding: 2rem;
        border-radius: 15px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(33,150,243,0.2);
    }
    
    .study-type-info {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 10px 10px 0;
    }
    
    .power-warning {
        background: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 10px 10px 0;
    }
    
    /* Fix tab title positioning and remove shadows */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        box-shadow: none !important;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
        box-shadow: none !important;
    }
    
    /* Fix visualization container title positioning */
    .visualization-container h3 {
        margin-top: 0;
        padding-top: 0;
    }
    
    /* Ensure consistent spacing for synchronized inputs */
    .sync-input-container {
        margin-bottom: 1rem;
    }
    
    .sync-input-label {
        font-weight: bold;
        margin-bottom: 0.5rem;
        display: block;
    }
</style>
""", unsafe_allow_html=True)

class SampleSizeCalculator:
    """Complete statistical sample size calculator with all ClinCalc functionality"""
    
    @staticmethod
    def z_score(alpha, two_sided=True):
        """Calculate z-score for given alpha level"""
        if two_sided:
            return stats.norm.ppf(1 - alpha/2)
        else:
            return stats.norm.ppf(1 - alpha)
    
    @staticmethod
    def calculate_continuous_two_groups(mean1, mean2, std_dev1, std_dev2, alpha=0.05, power=0.80, 
                                      allocation_ratio=1.0, two_sided=True, dropout_rate=0.0):
        """Calculate sample size for continuous outcomes, two independent groups with separate SDs"""
        
        # Calculate pooled standard deviation for effect size calculation
        pooled_std = math.sqrt((std_dev1**2 + std_dev2**2) / 2)
        
        # Calculate effect size using pooled SD
        effect_size = abs(mean1 - mean2) / pooled_std
        
        # Z-scores
        z_alpha = SampleSizeCalculator.z_score(alpha, two_sided)
        z_beta = stats.norm.ppf(power)
        
        # Sample size calculation using separate variances
        variance_term = std_dev1**2 + std_dev2**2 / allocation_ratio
        n1 = ((z_alpha + z_beta) ** 2 * variance_term) / ((mean1 - mean2) ** 2)
        
        # Store unadjusted values
        n1_unadjusted = math.ceil(n1)
        n2_unadjusted = math.ceil(n1 * allocation_ratio)
        total_unadjusted = n1_unadjusted + n2_unadjusted
        
        # Adjust for dropout
        if dropout_rate > 0:
            n1 = n1 / (1 - dropout_rate)
        
        n2 = n1 * allocation_ratio
        
        return {
            'n1': math.ceil(n1),
            'n2': math.ceil(n2),
            'total': math.ceil(n1 + n2),
            'n1_unadjusted': n1_unadjusted,
            'n2_unadjusted': n2_unadjusted,
            'total_unadjusted': total_unadjusted,
            'effect_size': effect_size,
            'pooled_std': pooled_std,
            'std_dev1': std_dev1,
            'std_dev2': std_dev2,
            'z_alpha': z_alpha,
            'z_beta': z_beta,
            'dropout_rate': dropout_rate
        }
    
    @staticmethod
    def calculate_continuous_one_group(sample_mean, population_mean, std_dev, alpha=0.05, 
                                     power=0.80, two_sided=True, dropout_rate=0.0):
        """Calculate sample size for continuous outcomes, one group vs population"""
        
        # Calculate effect size
        effect_size = abs(sample_mean - population_mean) / std_dev
        
        # Z-scores
        z_alpha = SampleSizeCalculator.z_score(alpha, two_sided)
        z_beta = stats.norm.ppf(power)
        
        # Sample size calculation
        n = ((z_alpha + z_beta) ** 2 * (std_dev ** 2)) / ((sample_mean - population_mean) ** 2)
        
        # Store unadjusted value
        n_unadjusted = math.ceil(n)
        
        # Adjust for dropout
        if dropout_rate > 0:
            n = n / (1 - dropout_rate)
        
        return {
            'n': math.ceil(n),
            'n_unadjusted': n_unadjusted,
            'effect_size': effect_size,
            'z_alpha': z_alpha,
            'z_beta': z_beta,
            'dropout_rate': dropout_rate
        }
    
    @staticmethod
    def calculate_proportions_two_groups(p1, p2, alpha=0.05, power=0.80, 
                                       allocation_ratio=1.0, two_sided=True, 
                                       continuity_correction=True, dropout_rate=0.0):
        """Calculate sample size for dichotomous outcomes, two independent groups"""
        
        # Calculate pooled proportion
        p_pooled = (p1 + allocation_ratio * p2) / (1 + allocation_ratio)
        q_pooled = 1 - p_pooled
        
        # Z-scores
        z_alpha = SampleSizeCalculator.z_score(alpha, two_sided)
        z_beta = stats.norm.ppf(power)
        
        # Variance calculations
        var_null = p_pooled * q_pooled * (1 + 1/allocation_ratio)
        var_alt = p1*(1-p1) + (p2*(1-p2))/allocation_ratio
        
        # Sample size calculation
        n1 = ((z_alpha * math.sqrt(var_null) + z_beta * math.sqrt(var_alt)) ** 2) / ((p1 - p2) ** 2)
        
        # Continuity correction (Yates correction)
        if continuity_correction:
            correction = 1 + 1/(2*n1*abs(p1 - p2))
            n1 = n1 * correction
        
        # Store unadjusted values
        n1_unadjusted = math.ceil(n1)
        n2_unadjusted = math.ceil(n1 * allocation_ratio)
        total_unadjusted = n1_unadjusted + n2_unadjusted
        
        # Adjust for dropout
        if dropout_rate > 0:
            n1 = n1 / (1 - dropout_rate)
        
        n2 = n1 * allocation_ratio
        
        return {
            'n1': math.ceil(n1),
            'n2': math.ceil(n2),
            'total': math.ceil(n1 + n2),
            'n1_unadjusted': n1_unadjusted,
            'n2_unadjusted': n2_unadjusted,
            'total_unadjusted': total_unadjusted,
            'p_pooled': p_pooled,
            'z_alpha': z_alpha,
            'z_beta': z_beta,
            'effect_size': abs(p1 - p2),
            'dropout_rate': dropout_rate
        }
    
    @staticmethod
    def calculate_proportions_one_group(sample_prop, population_prop, alpha=0.05, 
                                      power=0.80, two_sided=True, 
                                      continuity_correction=True, dropout_rate=0.0):
        """Calculate sample size for dichotomous outcomes, one group vs population"""
        
        # Z-scores
        z_alpha = SampleSizeCalculator.z_score(alpha, two_sided)
        z_beta = stats.norm.ppf(power)
        
        # q values
        q0 = 1 - population_prop
        q1 = 1 - sample_prop
        
        # Correct sample size calculation for one-sample proportion test
        numerator = (z_alpha * math.sqrt(population_prop * q0) + z_beta * math.sqrt(sample_prop * q1)) ** 2
        denominator = (sample_prop - population_prop) ** 2
        
        n = numerator / denominator
        
        # Continuity correction (Yates correction)
        if continuity_correction:
            correction = 1 + 1/(2*n*abs(sample_prop - population_prop))
            n = n * correction
        
        # Store unadjusted value
        n_unadjusted = math.ceil(n)
        
        # Adjust for dropout
        if dropout_rate > 0:
            n = n / (1 - dropout_rate)
        
        return {
            'n': math.ceil(n),
            'n_unadjusted': n_unadjusted,
            'z_alpha': z_alpha,
            'z_beta': z_beta,
            'effect_size': abs(sample_prop - population_prop),
            'p0': population_prop,
            'p1': sample_prop,
            'q0': q0,
            'q1': q1,
            'dropout_rate': dropout_rate
        }

class PostHocPowerAnalyzer:
    """Calculate post-hoc statistical power from actual study results"""
    
    @staticmethod
    def calculate_power_two_proportions(n1, n2, p1, p2, alpha=0.05, two_sided=True):
        """Calculate post-hoc power for two-proportion test"""
        
        # Pooled proportion
        p_pooled = (n1 * p1 + n2 * p2) / (n1 + n2)
        q_pooled = 1 - p_pooled
        
        # Standard errors
        se_null = math.sqrt(p_pooled * q_pooled * (1/n1 + 1/n2))
        se_alt = math.sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)
        
        # Effect size
        effect_size = abs(p1 - p2)
        
        # Critical value
        z_alpha = stats.norm.ppf(1 - alpha/2) if two_sided else stats.norm.ppf(1 - alpha)
        
        # Power calculation
        z_score = (effect_size - z_alpha * se_null) / se_alt
        power = stats.norm.cdf(z_score)
        
        return {
            'power': power,
            'power_percent': power * 100,
            'effect_size': effect_size,
            'se_null': se_null,
            'se_alt': se_alt,
            'z_alpha': z_alpha,
            'z_score': z_score,
            'p_pooled': p_pooled
        }
    
    @staticmethod
    def calculate_power_one_proportion(n, sample_prop, population_prop, alpha=0.05, two_sided=True):
        """Calculate post-hoc power for one-sample proportion test"""
        
        # Standard errors
        se_null = math.sqrt(population_prop * (1-population_prop) / n)
        se_alt = math.sqrt(sample_prop * (1-sample_prop) / n)
        
        # Effect size
        effect_size = abs(sample_prop - population_prop)
        
        # Critical value
        z_alpha = stats.norm.ppf(1 - alpha/2) if two_sided else stats.norm.ppf(1 - alpha)
        
        # Power calculation
        z_score = (effect_size - z_alpha * se_null) / se_alt
        power = stats.norm.cdf(z_score)
        
        return {
            'power': power,
            'power_percent': power * 100,
            'effect_size': effect_size,
            'se_null': se_null,
            'se_alt': se_alt,
            'z_alpha': z_alpha,
            'z_score': z_score
        }
    
    @staticmethod
    def calculate_power_two_means(n1, n2, mean1, mean2, std_dev, alpha=0.05, two_sided=True):
        """Calculate post-hoc power for two-sample t-test"""
        
        # Effect size (Cohen's d)
        effect_size = abs(mean1 - mean2) / std_dev
        
        # Standard error of difference
        se_diff = std_dev * math.sqrt(1/n1 + 1/n2)
        
        # Degrees of freedom
        df = n1 + n2 - 2
        
        # Critical value
        t_alpha = stats.t.ppf(1 - alpha/2, df) if two_sided else stats.t.ppf(1 - alpha, df)
        
        # Non-centrality parameter
        delta = abs(mean1 - mean2) / se_diff
        
        # Power calculation using non-central t-distribution
        power = 1 - stats.nct.cdf(t_alpha, df, delta)
        
        return {
            'power': power,
            'power_percent': power * 100,
            'effect_size': effect_size,
            'se_diff': se_diff,
            't_alpha': t_alpha,
            'delta': delta,
            'df': df
        }
    
    @staticmethod
    def calculate_power_one_mean(n, sample_mean, population_mean, std_dev, alpha=0.05, two_sided=True):
        """Calculate post-hoc power for one-sample t-test"""
        
        # Effect size (Cohen's d)
        effect_size = abs(sample_mean - population_mean) / std_dev
        
        # Standard error
        se = std_dev / math.sqrt(n)
        
        # Degrees of freedom
        df = n - 1
        
        # Critical value
        t_alpha = stats.t.ppf(1 - alpha/2, df) if two_sided else stats.t.ppf(1 - alpha, df)
        
        # Non-centrality parameter
        delta = abs(sample_mean - population_mean) / se
        
        # Power calculation
        power = 1 - stats.nct.cdf(t_alpha, df, delta)
        
        return {
            'power': power,
            'power_percent': power * 100,
            'effect_size': effect_size,
            'se': se,
            't_alpha': t_alpha,
            'delta': delta,
            'df': df
        }

def create_synchronized_input(label, min_val, max_val, default_val, step=0.01, help_text="", key_suffix="", format_str=None):
    """Create synchronized slider and number input that update each other in real-time"""
    
    # Create unique keys
    slider_key = f"slider_{key_suffix}_{hash(label)}_{id(label)}"
    number_key = f"number_{key_suffix}_{hash(label)}_{id(label)}"
    
    # Initialize with default if keys don't exist
    if slider_key not in st.session_state:
        st.session_state[slider_key] = float(default_val)
    if number_key not in st.session_state:
        st.session_state[number_key] = float(default_val)
    
    # Create the synchronized input container
    st.markdown(f'<div class="sync-input-container">', unsafe_allow_html=True)
    st.markdown(f'<div class="sync-input-label">{label}</div>', unsafe_allow_html=True)
    
    if help_text:
        st.caption(help_text)
    
    # Create columns for slider and number input
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Use on_change callbacks to keep them synchronized
        def sync_from_slider():
            st.session_state[number_key] = st.session_state[slider_key]
        
        slider_val = st.slider(
            "",
            min_value=float(min_val),
            max_value=float(max_val),
            value=st.session_state[slider_key],
            step=float(step),
            key=slider_key,
            label_visibility="collapsed",
            on_change=sync_from_slider
        )
    
    with col2:
        def sync_from_number():
            st.session_state[slider_key] = st.session_state[number_key]
        
        number_val = st.number_input(
            "",
            min_value=float(min_val),
            max_value=float(max_val),
            value=st.session_state[number_key],
            step=float(step),
            key=number_key,
            format=format_str,
            label_visibility="collapsed",
            on_change=sync_from_number
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Return the current value (both should be the same)
    return st.session_state[slider_key]

def display_study_type_info(study_design, outcome_type):
    """Display information about selected study types"""
    
    if study_design == "Two independent study groups":
        st.markdown("""
        <div class="study-type-info">
        <strong>Study Design: Two Independent Groups</strong><br>
        Two study groups will each receive different treatments. This design compares outcomes between two separate groups of participants.
        <br><em>Example: Comparing a new drug vs. placebo, or Treatment A vs. Treatment B</em>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="study-type-info">
        <strong>Study Design: One Group vs. Population</strong><br>
        One study cohort will be compared to a known value published in previous literature or established population parameters.
        <br><em>Example: Comparing your patient population to published national averages</em>
        </div>
        """, unsafe_allow_html=True)
    
    if outcome_type == "Dichotomous (yes/no)":
        st.markdown("""
        <div class="study-type-info">
        <strong>Outcome Type: Dichotomous (Binomial)</strong><br>
        The primary endpoint is <strong>binomial</strong> - only two possible outcomes.
        <br><em>Examples: mortality (dead/not dead), pregnant (pregnant/not), response to treatment (yes/no)</em>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="study-type-info">
        <strong>Outcome Type: Continuous (Average)</strong><br>
        The primary endpoint is an <strong>average</strong> - measured on a continuous scale.
        <br><em>Examples: blood pressure reduction (mmHg), weight loss (kg), pain score reduction</em>
        </div>
        """, unsafe_allow_html=True)

def create_enhanced_visualizations(study_design, outcome_type, base_params, results):
    """Create enhanced parameter sensitivity analysis charts"""
    
    st.markdown(
        """
        <div class="visualization-container">
            <h3>📊 <b>Interactive Parameter Sensitivity Analysis</b></h3>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Create tabs for different visualization types
    tab1, tab2, tab3 = st.tabs(["📈 Sensitivity Curves", "🎯 Power Analysis", "📊 Comparison Charts"])
    
    with tab1:
        # Effect Size vs Sample Size
        fig1 = go.Figure()
        
        if study_design == "Two independent study groups" and outcome_type == "Continuous (means)":
            effect_sizes = np.linspace(0.2, 2.0, 30)
            sample_sizes = []
            
            # Get standard deviation - check for pooled_std first, then fall back to std_dev
            std_dev_to_use = base_params.get('pooled_std', base_params.get('std_dev', 2.0))
            std_dev1 = base_params.get('std_dev1', std_dev_to_use)
            std_dev2 = base_params.get('std_dev2', std_dev_to_use)
            
            for es in effect_sizes:
                mean_diff = es * std_dev_to_use
                try:
                    result = SampleSizeCalculator.calculate_continuous_two_groups(
                        base_params['mean1'], 
                        base_params['mean1'] + mean_diff,
                        std_dev1,
                        std_dev2,
                        base_params['alpha'],
                        base_params['power'],
                        dropout_rate=0.0  # Use unadjusted for sensitivity analysis
                    )
                    sample_sizes.append(result['total_unadjusted'])
                except:
                    sample_sizes.append(np.nan)
            
            fig1.add_trace(go.Scatter(
                x=effect_sizes, y=sample_sizes, 
                mode='lines+markers',
                name='Effect Size vs Sample Size',
                line=dict(color='#2E86AB', width=4),
                marker=dict(size=8)
            ))
            
            # Add marker for current study
            current_effect_size = results.get('effect_size', 0)
            current_sample_size = results.get('total_unadjusted', results.get('total', 0))
            
            fig1.add_trace(go.Scatter(
                x=[current_effect_size],
                y=[current_sample_size],
                mode='markers',
                name='Your Study Design',
                marker=dict(size=15, color='red', symbol='star', 
                           line=dict(width=2, color='white'))
            ))
            
            fig1.update_layout(
                title="Effect Size vs Required Sample Size",
                xaxis_title="Effect Size (Cohen's d)",
                yaxis_title="Total Sample Size Required",
                height=500,
                showlegend=True,
                font=dict(size=14)
            )
            
        elif study_design == "Two independent study groups" and outcome_type == "Dichotomous (yes/no)":
            # Proportion difference vs sample size
            p1_base = base_params.get('p1', 0.3)
            p2_range = np.linspace(0.05, 0.95, 30)
            sample_sizes = []
            
            for p2 in p2_range:
                if abs(p1_base - p2) > 0.01:
                    try:
                        result = SampleSizeCalculator.calculate_proportions_two_groups(
                            p1_base, p2, base_params['alpha'], base_params['power'],
                            dropout_rate=0.0  # Use unadjusted for sensitivity analysis
                        )
                        sample_sizes.append(result['total_unadjusted'])
                    except:
                        sample_sizes.append(np.nan)
                else:
                    sample_sizes.append(np.nan)
            
            fig1.add_trace(go.Scatter(
                x=np.abs(p2_range - p1_base), y=sample_sizes,
                mode='lines+markers',
                name='Effect Size vs Sample Size',
                line=dict(color='#2E86AB', width=4),
                marker=dict(size=8)
            ))
            
            # Add marker for current study
            current_effect_size = results.get('effect_size', 0)
            current_sample_size = results.get('total_unadjusted', results.get('total', 0))
            
            fig1.add_trace(go.Scatter(
                x=[current_effect_size],
                y=[current_sample_size],
                mode='markers',
                name='Your Study Design',
                marker=dict(size=15, color='red', symbol='star',
                           line=dict(width=2, color='white'))
            ))
            
            fig1.update_layout(
                title="Effect Size vs Required Sample Size",
                xaxis_title="Effect Size (Absolute Difference in Proportions)",
                yaxis_title="Total Sample Size Required",
                height=500,
                showlegend=True,
                font=dict(size=14)
            )
            
        # Similar updates for one group studies...
        
        st.plotly_chart(fig1, use_container_width=True)
    
    with tab2:
        # Power Analysis
        powers = np.linspace(0.70, 0.95, 20)
        power_sample_sizes = []
        
        for power in powers:
            try:
                if study_design == "Two independent study groups":
                    if outcome_type == "Continuous (means)":
                        # Get standard deviations with fallback
                        std_dev_to_use = base_params.get('pooled_std', base_params.get('std_dev', 2.0))
                        std_dev1 = base_params.get('std_dev1', std_dev_to_use)
                        std_dev2 = base_params.get('std_dev2', std_dev_to_use)
                        
                        result = SampleSizeCalculator.calculate_continuous_two_groups(
                            base_params['mean1'], base_params['mean2'],
                            std_dev1, std_dev2, 
                            base_params['alpha'], power,
                            dropout_rate=0.0  # Use unadjusted for analysis
                        )
                        power_sample_sizes.append(result['total_unadjusted'])
                    else:  # Dichotomous
                        result = SampleSizeCalculator.calculate_proportions_two_groups(
                            base_params['p1'], base_params['p2'],
                            base_params['alpha'], power,
                            dropout_rate=0.0  # Use unadjusted for analysis
                        )
                        power_sample_sizes.append(result['total_unadjusted'])
                else:  # One group vs population
                    if outcome_type == "Continuous (means)":
                        result = SampleSizeCalculator.calculate_continuous_one_group(
                            base_params['sample_mean'], base_params['population_mean'],
                            base_params['std_dev'], base_params['alpha'], power,
                            dropout_rate=0.0  # Use unadjusted for analysis
                        )
                        power_sample_sizes.append(result['n_unadjusted'])
                    else:  # Dichotomous
                        result = SampleSizeCalculator.calculate_proportions_one_group(
                            base_params['sample_prop'], base_params['population_prop'],
                            base_params['alpha'], power,
                            dropout_rate=0.0  # Use unadjusted for analysis
                        )
                        power_sample_sizes.append(result['n_unadjusted'])
            except:
                power_sample_sizes.append(np.nan)
        
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=powers, y=power_sample_sizes,
            mode='lines+markers',
            name='Power vs Sample Size',
            line=dict(color='#A23B72', width=4),
            marker=dict(size=8)
        ))
        
        # Add marker for current study
        current_power = base_params.get('power', 0.8)
        current_sample = results.get('total_unadjusted', results.get('total', results.get('n_unadjusted', results.get('n', 0))))
        
        fig2.add_trace(go.Scatter(
            x=[current_power],
            y=[current_sample],
            mode='markers',
            name='Your Study Design',
            marker=dict(size=15, color='red', symbol='star',
                       line=dict(width=2, color='white'))
        ))
        
        # Add horizontal line at current study sample size
        fig2.add_hline(y=current_sample, line_dash="dash", line_color="red", 
                       annotation_text=f"Your Study Sample Size = {current_sample}")
        
        # Add vertical line at current study power
        fig2.add_vline(x=current_power, line_dash="dash", line_color="blue", 
                       annotation_text=f"Your Study Power = {current_power*100:.0f}%")
        
        fig2.update_layout(
            title="Statistical Power vs Required Sample Size",
            xaxis_title="Statistical Power (1-β)",
            yaxis_title="Sample Size Required",
            height=500,
            showlegend=True,
            font=dict(size=14)
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab3:
        # Alpha comparison
        alphas = [0.01, 0.05, 0.10]
        alpha_sample_sizes = []
        alpha_labels = ['α = 0.01', 'α = 0.05', 'α = 0.10']
        
        for alpha in alphas:
            try:
                if study_design == "Two independent study groups":
                    if outcome_type == "Continuous (means)":
                        # Get standard deviations with fallback
                        std_dev_to_use = base_params.get('pooled_std', base_params.get('std_dev', 2.0))
                        std_dev1 = base_params.get('std_dev1', std_dev_to_use)
                        std_dev2 = base_params.get('std_dev2', std_dev_to_use)
                        
                        result = SampleSizeCalculator.calculate_continuous_two_groups(
                            base_params['mean1'], base_params['mean2'],
                            std_dev1, std_dev2,
                            alpha, base_params['power'],
                            dropout_rate=0.0  # Use unadjusted for analysis
                        )
                        alpha_sample_sizes.append(result['total_unadjusted'])
                    else:  # Dichotomous
                        result = SampleSizeCalculator.calculate_proportions_two_groups(
                            base_params['p1'], base_params['p2'],
                            alpha, base_params['power'],
                            dropout_rate=0.0  # Use unadjusted for analysis
                        )
                        alpha_sample_sizes.append(result['total_unadjusted'])
                else:  # One group vs population
                    if outcome_type == "Continuous (means)":
                        result = SampleSizeCalculator.calculate_continuous_one_group(
                            base_params['sample_mean'], base_params['population_mean'],
                            base_params['std_dev'], alpha, base_params['power'],
                            dropout_rate=0.0  # Use unadjusted for analysis
                        )
                        alpha_sample_sizes.append(result['n_unadjusted'])
                    else:  # Dichotomous
                        result = SampleSizeCalculator.calculate_proportions_one_group(
                            base_params['sample_prop'], base_params['population_prop'],
                            alpha, base_params['power'],
                            dropout_rate=0.0  # Use unadjusted for analysis
                        )
                        alpha_sample_sizes.append(result['n_unadjusted'])
            except:
                alpha_sample_sizes.append(np.nan)
        
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=alpha_labels,
            y=alpha_sample_sizes,
            name='Alpha Levels',
            marker_color=['#e74c3c', '#f39c12', '#27ae60'],
            text=[f'{int(size)}' for size in alpha_sample_sizes if not np.isnan(size)],
            textposition='auto',
        ))
        
        fig3.update_layout(
            title="Alpha Level Impact on Sample Size",
            xaxis_title="Alpha Level (Type I Error Rate)",
            yaxis_title="Sample Size Required",
            height=500,
            showlegend=False,
            font=dict(size=14)
        )
        
        st.plotly_chart(fig3, use_container_width=True)
        
        # Additional comparison chart: Power levels
        st.markdown("#### **Power Level Comparison**")
        
        power_levels = [0.70, 0.80, 0.90, 0.95]
        power_sample_sizes_comp = []
        power_labels = ['70%', '80%', '90%', '95%']
        
        for power in power_levels:
            try:
                if study_design == "Two independent study groups":
                    if outcome_type == "Continuous (means)":
                        # Get standard deviations with fallback
                        std_dev_to_use = base_params.get('pooled_std', base_params.get('std_dev', 2.0))
                        std_dev1 = base_params.get('std_dev1', std_dev_to_use)
                        std_dev2 = base_params.get('std_dev2', std_dev_to_use)
                        
                        result = SampleSizeCalculator.calculate_continuous_two_groups(
                            base_params['mean1'], base_params['mean2'],
                            std_dev1, std_dev2,
                            base_params['alpha'], power,
                            dropout_rate=0.0  # Use unadjusted for analysis
                        )
                        power_sample_sizes_comp.append(result['total_unadjusted'])
                    else:  # Dichotomous
                        result = SampleSizeCalculator.calculate_proportions_two_groups(
                            base_params['p1'], base_params['p2'],
                            base_params['alpha'], power,
                            dropout_rate=0.0  # Use unadjusted for analysis
                        )
                        power_sample_sizes_comp.append(result['total_unadjusted'])
                else:  # One group vs population
                    if outcome_type == "Continuous (means)":
                        result = SampleSizeCalculator.calculate_continuous_one_group(
                            base_params['sample_mean'], base_params['population_mean'],
                            base_params['std_dev'], base_params['alpha'], power,
                            dropout_rate=0.0  # Use unadjusted for analysis
                        )
                        power_sample_sizes_comp.append(result['n_unadjusted'])
                    else:  # Dichotomous
                        result = SampleSizeCalculator.calculate_proportions_one_group(
                            base_params['sample_prop'], base_params['population_prop'],
                            base_params['alpha'], power,
                            dropout_rate=0.0  # Use unadjusted for analysis
                        )
                        power_sample_sizes_comp.append(result['n_unadjusted'])
            except:
                power_sample_sizes_comp.append(np.nan)
        
        fig4 = go.Figure()
        fig4.add_trace(go.Bar(
            x=power_labels,
            y=power_sample_sizes_comp,
            name='Power Levels',
            marker_color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c'],
            text=[f'{int(size)}' for size in power_sample_sizes_comp if not np.isnan(size)],
            textposition='auto',
        ))
        
        fig4.update_layout(
            title="Statistical Power Impact on Sample Size",
            xaxis_title="Statistical Power Level",
            yaxis_title="Sample Size Required",
            height=500,
            showlegend=False,
            font=dict(size=14)
        )
        
        st.plotly_chart(fig4, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_latex_formula_detailed(study_design, outcome_type, params, is_posthoc=False):
    """Display detailed LaTeX formulas with comprehensive explanations and numeric substitution"""
    
    st.markdown('<div class="formula-container">', unsafe_allow_html=True)
    
    if not is_posthoc:
        st.markdown("### 🧮 **Sample Size Formula & Parameter Substitution**")
        
        # Show dropout rate explanation when applicable
        dropout_rate = params.get('dropout_rate', 0.0)
        if dropout_rate > 0:
            st.markdown(f"""
            **📊 Dropout Rate Adjustment Applied:**
            
            The formulas below show the base sample size calculation. Your results include a {dropout_rate*100:.0f}% adjustment for expected dropouts:
            
            - **Adjusted Sample Size** = Base Sample Size ÷ (1 - Dropout Rate)
            - **Adjustment Factor** = 1 ÷ (1 - {dropout_rate:.2f}) = {1/(1-dropout_rate):.2f}
            """)
    else:
        st.markdown("### 🔄 **Post-Hoc Power Formula & Parameter Substitution**")
    
    if study_design == "Two independent study groups":
        if outcome_type == "Continuous (means)":
            if not is_posthoc:
                st.markdown("#### **Two-Sample T-Test Sample Size Formula:**")
                st.latex(r'''n_1 = \frac{(z_{1-\alpha/2} + z_{1-\beta})^2 (\sigma_1^2 + \sigma_2^2/k)}{(\mu_1 - \mu_2)^2}''')
                
                # Show numeric substitution if parameters available
                if all(key in params for key in ['mean1', 'mean2', 'std_dev1', 'std_dev2', 'alpha', 'power']):
                    st.markdown("#### **Parameter Substitution:**")
                    
                    mean1 = params['mean1']
                    mean2 = params['mean2']
                    std1 = params['std_dev1']
                    std2 = params['std_dev2']
                    alpha = params['alpha']
                    power = params['power']
                    z_alpha = params.get('z_alpha', 1.96)
                    z_beta = params.get('z_beta', 0.84)
                    allocation_ratio = params.get('allocation_ratio', 1.0)
                    dropout_rate = params.get('dropout_rate', 0.0)
                    
                    st.markdown(f"""
                    **Given Parameters:**
                    - μ₁ = {mean1:.1f} (group 1 mean)
                    - μ₂ = {mean2:.1f} (group 2 mean)
                    - σ₁ = {std1:.1f} (group 1 standard deviation)
                    - σ₂ = {std2:.1f} (group 2 standard deviation)
                    - α = {alpha:.2f}, z₁₋α/₂ = {z_alpha:.2f}
                    - β = {1-power:.2f}, z₁₋β = {z_beta:.2f}
                    - k = {allocation_ratio:.1f} (allocation ratio)
                    """)
                    
                    variance_term = std1**2 + std2**2/allocation_ratio
                    numerator = (z_alpha + z_beta)**2 * variance_term
                    denominator = (mean1 - mean2)**2
                    n1_result = math.ceil(numerator / denominator)
                    
                    st.latex(f'''n_1 = \\frac{{({z_alpha:.2f} + {z_beta:.2f})^2 \\times ({std1:.1f}^2 + {std2:.1f}^2/{allocation_ratio:.1f})}}{{({mean1:.1f} - {mean2:.1f})^2}}''')
                    
                    st.latex(f'''n_1 = \\frac{{{(z_alpha + z_beta)**2:.2f} \\times {variance_term:.2f}}}{{{denominator:.2f}}} = {n1_result}''')
                    
                    base_total = math.ceil(n1_result * (1 + allocation_ratio))
                    st.markdown(f"**Base Sample Size:** n₁ = {n1_result}, n₂ = {math.ceil(n1_result * allocation_ratio)}, **Total = {base_total}**")
                    
                    if dropout_rate > 0:
                        adjusted_total = math.ceil(base_total / (1 - dropout_rate))
                        st.markdown(f"**Adjusted for {dropout_rate*100:.0f}% Dropout:** **Total = {adjusted_total}**")
                        
            else:
                st.markdown("#### **Two-Sample T-Test Post-Hoc Power Formula:**")
                st.latex(r'''Power = 1 - T_{df,\delta}(t_{1-\alpha/2})''')
                st.markdown("**where:**")
                st.latex(r'''\delta = \frac{|\bar{x}_1 - \bar{x}_2|}{s_p\sqrt{1/n_1 + 1/n_2}}''')
                st.latex(r'''s_p = \sqrt{\frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1 + n_2 - 2}}''')
                st.latex(r'''df = n_1 + n_2 - 2''')
                
                # Show numerical substitution for post-hoc power
                if all(key in params for key in ['n1', 'n2', 'mean1', 'mean2', 'std_dev']):
                    st.markdown("#### **Parameter Substitution:**")
                    
                    n1 = params['n1']
                    n2 = params['n2']
                    mean1 = params['mean1']
                    mean2 = params['mean2']
                    std_dev = params['std_dev']
                    alpha = params['alpha']
                    
                    df = n1 + n2 - 2
                    se_diff = std_dev * math.sqrt(1/n1 + 1/n2)
                    delta = abs(mean1 - mean2) / se_diff
                    t_alpha = stats.t.ppf(1 - alpha/2, df)
                    power = 1 - stats.nct.cdf(t_alpha, df, delta)
                    
                    st.markdown(f"""
                    **Given Parameters:**
                    - n₁ = {n1}, n₂ = {n2}
                    - x̄₁ = {mean1:.1f}, x̄₂ = {mean2:.1f}
                    - sp = {std_dev:.1f} (pooled standard deviation)
                    - α = {alpha:.2f}
                    """)
                    
                    st.latex(f'''\\delta = \\frac{{|{mean1:.1f} - {mean2:.1f}|}}{{{std_dev:.1f} \\times \\sqrt{{1/{n1} + 1/{n2}}}}} = \\frac{{{abs(mean1-mean2):.1f}}}{{{se_diff:.3f}}} = {delta:.3f}''')
                    
                    st.latex(f'''Power = 1 - T_{{{df},{delta:.3f}}}({t_alpha:.3f}) = {power:.3f} = {power*100:.1f}\\%''')
                
        else:  # Dichotomous two groups
            if not is_posthoc:
                st.markdown("#### **Two-Proportion Z-Test Sample Size Formula:**")
                st.latex(r'''n_1 = \frac{[z_{1-\alpha/2}\sqrt{\bar{p}\bar{q}(1 + \frac{1}{k})} + z_{1-\beta}\sqrt{p_1q_1 + \frac{p_2q_2}{k}}]^2}{(p_1 - p_2)^2}''')
                
                # Show numeric substitution for two proportions
                if all(key in params for key in ['p1', 'p2', 'alpha', 'power']):
                    st.markdown("#### **Parameter Substitution:**")
                    
                    p1 = params['p1']
                    p2 = params['p2']
                    q1 = 1 - p1
                    q2 = 1 - p2
                    alpha = params['alpha']
                    power = params['power']
                    z_alpha = params.get('z_alpha', 1.96)
                    z_beta = params.get('z_beta', 0.84)
                    allocation_ratio = params.get('allocation_ratio', 1.0)
                    p_pooled = params.get('p_pooled', (p1 + allocation_ratio * p2) / (1 + allocation_ratio))
                    q_pooled = 1 - p_pooled
                    dropout_rate = params.get('dropout_rate', 0.0)
                    
                    st.markdown(f"""
                    **Given Parameters:**
                    - p₁ = {p1:.2f}, q₁ = {q1:.2f} (group 1 proportions)
                    - p₂ = {p2:.2f}, q₂ = {q2:.2f} (group 2 proportions)
                    - p̄ = {p_pooled:.3f}, q̄ = {q_pooled:.3f} (pooled proportions)
                    - α = {alpha:.2f}, z₁₋α/₂ = {z_alpha:.2f}
                    - β = {1-power:.2f}, z₁₋β = {z_beta:.2f}
                    - k = {allocation_ratio:.1f} (allocation ratio)
                    """)
                    
                    var_null = p_pooled * q_pooled * (1 + 1/allocation_ratio)
                    var_alt = p1 * q1 + p2 * q2 / allocation_ratio
                    numerator = (z_alpha * math.sqrt(var_null) + z_beta * math.sqrt(var_alt))**2
                    denominator = (p1 - p2)**2
                    n1_result = math.ceil(numerator / denominator)
                    
                    st.latex(f'''n_1 = \\frac{{[{z_alpha:.2f}\\sqrt{{{p_pooled:.3f} \\times {q_pooled:.3f} \\times (1 + 1/{allocation_ratio:.1f})}} + {z_beta:.2f}\\sqrt{{{p1:.2f} \\times {q1:.2f} + {p2:.2f} \\times {q2:.2f}/{allocation_ratio:.1f}}}]^2}}{{({p1:.2f} - {p2:.2f})^2}}''')
                    
                    st.latex(f'''n_1 = \\frac{{[{z_alpha * math.sqrt(var_null):.3f} + {z_beta * math.sqrt(var_alt):.3f}]^2}}{{{denominator:.4f}}} = {n1_result}''')
                    
                    base_total = math.ceil(n1_result * (1 + allocation_ratio))
                    st.markdown(f"**Base Sample Size:** n₁ = {n1_result}, n₂ = {math.ceil(n1_result * allocation_ratio)}, **Total = {base_total}**")
                    
                    if dropout_rate > 0:
                        adjusted_total = math.ceil(base_total / (1 - dropout_rate))
                        st.markdown(f"**Adjusted for {dropout_rate*100:.0f}% Dropout:** **Total = {adjusted_total}**")
            else:
                st.markdown("#### **Two-Proportion Post-Hoc Power Formula:**")
                st.latex(r'''Power = \Phi\left(\frac{|\hat{p}_1 - \hat{p}_2| - z_{1-\alpha/2}\sqrt{\bar{p}\bar{q}(\frac{1}{n_1} + \frac{1}{n_2})}}{\sqrt{\frac{\hat{p}_1(1-\hat{p}_1)}{n_1} + \frac{\hat{p}_2(1-\hat{p}_2)}{n_2}}}\right)''')
                
                st.markdown("**where:**")
                st.latex(r'''\bar{p} = \frac{n_1\hat{p}_1 + n_2\hat{p}_2}{n_1 + n_2}''')
                st.latex(r'''\bar{q} = 1 - \bar{p}''')
                
                # Show parameter substitution
                if all(key in params for key in ['n1', 'n2', 'p1', 'p2', 'alpha']):
                    st.markdown("#### **Parameter Substitution:**")
                    
                    n1 = params['n1']
                    n2 = params['n2'] 
                    p1 = params['p1']
                    p2 = params['p2']
                    q1 = 1 - p1
                    q2 = 1 - p2
                    alpha = params['alpha']
                    delta = abs(p2 - p1)
                    z_alpha = 1.96 if alpha == 0.05 else stats.norm.ppf(1 - alpha/2)
                    
                    # Calculate pooled proportion from observed data
                    p_pooled = (n1 * p1 + n2 * p2) / (n1 + n2)
                    q_pooled = 1 - p_pooled
                    
                    # Calculate components step by step
                    numerator = delta
                    denominator = math.sqrt(p1*q1/n1 + p2*q2/n2)
                    first_term = numerator / denominator
                    
                    pooled_se = math.sqrt(p_pooled * q_pooled * (1/n1 + 1/n2))
                    second_term = z_alpha * pooled_se / denominator
                    
                    final_z = first_term - second_term
                    power = stats.norm.cdf(final_z)
                    
                    st.markdown(f"""
                    **Given Parameters:**
                    - n₁ = {n1}, n₂ = {n2}
                    - p̂₁ = {p1:.2f}, p̂₂ = {p2:.2f}
                    - p̄ = {p_pooled:.3f}, q̄ = {q_pooled:.3f}
                    - α = {alpha:.2f}, z₁₋α/₂ = {z_alpha:.2f}
                    """)
                    
                    st.latex(f'''Power = \\Phi\\left(\\frac{{|{p1:.2f} - {p2:.2f}| - {z_alpha:.2f} \\times \\sqrt{{{p_pooled:.3f} \\times {q_pooled:.3f} \\times (\\frac{{1}}{{{n1}}} + \\frac{{1}}{{{n2}}})}}}}{{\\sqrt{{\\frac{{{p1:.2f} \\times {q1:.2f}}}{{{n1}}} + \\frac{{{p2:.2f} \\times {q2:.2f}}}{{{n2}}}}}}}\\right)''')
                    
                    st.latex(f'''Power = \\Phi({final_z:.3f}) = {power:.3f} = {power*100:.1f}\\%''')
    
    else:  # One group vs population
        if outcome_type == "Continuous (means)":
            if not is_posthoc:
                st.markdown("#### **One-Sample T-Test Sample Size Formula:**")
                st.latex(r'''N = \frac{(z_{1-\alpha/2} + z_{1-\beta})^2 \cdot \sigma^2}{(\mu_{sample} - \mu_{population})^2}''')
                
                # Show numeric substitution for one-sample continuous
                if all(key in params for key in ['sample_mean', 'population_mean', 'std_dev', 'alpha', 'power']):
                    st.markdown("#### **Parameter Substitution:**")
                    
                    sample_mean = params['sample_mean']
                    pop_mean = params['population_mean']
                    std_dev = params['std_dev']
                    alpha = params['alpha']
                    power = params['power']
                    z_alpha = params.get('z_alpha', 1.96)
                    z_beta = params.get('z_beta', 0.84)
                    dropout_rate = params.get('dropout_rate', 0.0)
                    
                    numerator = (z_alpha + z_beta)**2 * std_dev**2
                    denominator = (sample_mean - pop_mean)**2
                    n_result = math.ceil(numerator / denominator)
                    
                    st.markdown(f"""
                    **Given Parameters:**
                    - μsample = {sample_mean:.1f} (expected sample mean)
                    - μpopulation = {pop_mean:.1f} (population mean)
                    - σ = {std_dev:.1f} (standard deviation)
                    - α = {alpha:.2f}, z₁₋α/₂ = {z_alpha:.2f}
                    - β = {1-power:.2f}, z₁₋β = {z_beta:.2f}
                    """)
                    
                    st.latex(f'''N = \\frac{{({z_alpha:.2f} + {z_beta:.2f})^2 \\times {std_dev:.1f}^2}}{{({sample_mean:.1f} - {pop_mean:.1f})^2}}''')
                    
                    st.latex(f'''N = \\frac{{{numerator:.2f}}}{{{denominator:.2f}}} = {n_result}''')
                    
                    if dropout_rate > 0:
                        adjusted_n = math.ceil(n_result / (1 - dropout_rate))
                        st.markdown(f"**Base Sample Size:** N = {n_result}")
                        st.markdown(f"**Adjusted for {dropout_rate*100:.0f}% Dropout:** **N = {adjusted_n}**")
                    
            else:
                st.markdown("#### **One-Sample T-Test Post-Hoc Power Formula:**")
                st.latex(r'''Power = 1 - T_{df,\delta}(t_{1-\alpha/2})''')
                st.markdown("**where:**")
                st.latex(r'''\delta = \frac{|\bar{x} - \mu_0|}{s/\sqrt{n}}''')
                st.latex(r'''df = n - 1''')
                
                # Show parameter substitution for post-hoc one-sample continuous
                if all(key in params for key in ['n', 'sample_mean', 'population_mean', 'std_dev', 'alpha']):
                    st.markdown("#### **Parameter Substitution:**")
                    
                    n = params['n']
                    sample_mean = params['sample_mean']
                    pop_mean = params['population_mean']
                    std_dev = params['std_dev']
                    alpha = params['alpha']
                    
                    df = n - 1
                    se = std_dev / math.sqrt(n)
                    delta = abs(sample_mean - pop_mean) / se
                    t_alpha = stats.t.ppf(1 - alpha/2, df)
                    power = 1 - stats.nct.cdf(t_alpha, df, delta)
                    
                    st.markdown(f"""
                    **Given Parameters:**
                    - n = {n} (sample size)
                    - x̄ = {sample_mean:.1f} (observed sample mean)
                    - μ₀ = {pop_mean:.1f} (population mean)
                    - s = {std_dev:.1f} (standard deviation)
                    - α = {alpha:.2f}
                    """)
                    
                    st.latex(f'''\\delta = \\frac{{|{sample_mean:.1f} - {pop_mean:.1f}|}}{{{std_dev:.1f}/\\sqrt{{{n}}}}} = \\frac{{{abs(sample_mean - pop_mean):.1f}}}{{{se:.3f}}} = {delta:.3f}''')
                    
                    st.latex(f'''Power = 1 - T_{{{df},{delta:.3f}}}({t_alpha:.3f}) = {power:.3f} = {power*100:.1f}\\%''')
                
        else:  # Dichotomous one group  
            if not is_posthoc:
                st.markdown("#### **One-Sample Proportion Test Sample Size Formula:**")
                st.latex(r'''N = \frac{[z_{1-\alpha/2}\sqrt{p_0q_0} + z_{1-\beta}\sqrt{p_1q_1}]^2}{(p_1 - p_0)^2}''')
                
                # Show numeric substitution for one-sample proportion
                if all(key in params for key in ['sample_prop', 'population_prop', 'alpha', 'power']):
                    st.markdown("#### **Parameter Substitution:**")
                    
                    p0 = params['population_prop']
                    p1 = params['sample_prop'] 
                    q0 = 1 - p0
                    q1 = 1 - p1
                    alpha = params['alpha']
                    power = params['power']
                    z_alpha = params.get('z_alpha', 1.96)
                    z_beta = params.get('z_beta', 0.84)
                    dropout_rate = params.get('dropout_rate', 0.0)
                    
                    numerator_part1 = z_alpha * math.sqrt(p0 * q0)
                    numerator_part2 = z_beta * math.sqrt(p1 * q1)
                    numerator_total = numerator_part1 + numerator_part2
                    denominator = (p1 - p0)**2
                    n_result = math.ceil((numerator_total**2) / denominator)
                    
                    st.markdown(f"""
                    **Given Parameters:**
                    - p₀ = {p0:.2f} (population proportion)
                    - p₁ = {p1:.2f} (study proportion)  
                    - q₀ = 1 - p₀ = {q0:.2f}
                    - q₁ = 1 - p₁ = {q1:.2f}
                    - α = {alpha:.2f}, z₁₋α/₂ = {z_alpha:.2f}
                    - β = {1-power:.2f}, z₁₋β = {z_beta:.2f}
                    """)
                    
                    st.latex(f'''N = \\frac{{[{z_alpha:.2f} \\times \\sqrt{{{p0:.2f} \\times {q0:.2f}}} + {z_beta:.2f} \\times \\sqrt{{{p1:.2f} \\times {q1:.2f}}}]^2}}{{({p1:.2f} - {p0:.2f})^2}}''')
                    
                    st.latex(f'''N = \\frac{{[{numerator_part1:.3f} + {numerator_part2:.3f}]^2}}{{{denominator:.4f}}} = {n_result}''')
                    
                    if dropout_rate > 0:
                        adjusted_n = math.ceil(n_result / (1 - dropout_rate))
                        st.markdown(f"**Base Sample Size:** N = {n_result}")
                        st.markdown(f"**Adjusted for {dropout_rate*100:.0f}% Dropout:** **N = {adjusted_n}**")
                    
            else:
                st.markdown("#### **One-Sample Proportion Post-Hoc Power Formula:**")
                st.latex(r'''Power = \Phi\left(\frac{|\hat{p} - p_0| - z_{1-\alpha/2}\sqrt{p_0q_0/n}}{\sqrt{\hat{p}(1-\hat{p})/n}}\right)''')
                
                # Show parameter substitution for post-hoc one proportion
                if all(key in params for key in ['n', 'sample_prop', 'population_prop', 'alpha']):
                    st.markdown("#### **Parameter Substitution:**")
                    
                    n = params['n']
                    p_hat = params['sample_prop']
                    p0 = params['population_prop']
                    q0 = 1 - p0
                    q_hat = 1 - p_hat
                    alpha = params['alpha']
                    z_alpha = 1.96 if alpha == 0.05 else stats.norm.ppf(1 - alpha/2)
                    
                    delta = abs(p_hat - p0)
                    se_null = math.sqrt(p0 * q0 / n)
                    se_alt = math.sqrt(p_hat * q_hat / n)
                    
                    z_score = (delta - z_alpha * se_null) / se_alt
                    power = stats.norm.cdf(z_score)
                    
                    st.markdown(f"""
                    **Given Parameters:**
                    - n = {n} (sample size)
                    - p̂ = {p_hat:.2f} (observed proportion)
                    - p₀ = {p0:.2f} (population proportion)
                    - α = {alpha:.2f}, z₁₋α/₂ = {z_alpha:.2f}
                    """)
                    
                    st.latex(f'''Power = \\Phi\\left(\\frac{{|{p_hat:.2f} - {p0:.2f}| - {z_alpha:.2f}\\sqrt{{{p0:.2f} \\times {q0:.2f}/{n}}}}}{{\\sqrt{{{p_hat:.2f} \\times {q_hat:.2f}/{n}}}}}\\right)''')
                    
                    st.latex(f'''Power = \\Phi({z_score:.3f}) = {power:.3f} = {power*100:.1f}\\%''')

def display_professional_results_tables(results, study_design, outcome_type, params, is_posthoc=False):
    """Display results using proper Streamlit tables"""
    
    # Results header
    if is_posthoc:
        st.markdown('<div class="results-header">POST-HOC POWER ANALYSIS RESULTS</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="results-header">SAMPLE SIZE CALCULATION RESULTS</div>', unsafe_allow_html=True)
    
    # Study type title
    if study_design == "Two independent study groups":
        if outcome_type == "Continuous (means)":
            title = "Continuous Endpoint, Two-Sample Study"
        else:
            title = "Dichotomous Endpoint, Two-Sample Study"
    else:
        if outcome_type == "Continuous (means)":
            title = "Continuous Endpoint, One-Sample Study"
        else:
            title = "Dichotomous Endpoint, One-Sample Study"
    
    st.markdown(f'<div class="results-title">{title}</div>', unsafe_allow_html=True)
    
    # Dropout rate warning if applicable
    dropout_rate = params.get('dropout_rate', 0.0)
    if not is_posthoc and dropout_rate > 0:
        st.markdown(f"""
        <div style="background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 1rem; margin: 1rem 0; border-radius: 0 10px 10px 0;">
        <strong>📊 Dropout Adjustment Applied:</strong><br>
        Sample sizes below include {dropout_rate*100:.0f}% expected dropout rate. 
        The final sample size accounts for participants who may withdraw during the study.
        </div>
        """, unsafe_allow_html=True)
    
    # Main results display
    col1, col2 = st.columns(2)
    
    with col1:
        if is_posthoc:
            st.markdown("### **Calculated Power**")
            power_val = results.get('power_percent', results.get('power', 0) * 100)
            st.markdown(f'<div class="metric-card"><div class="metric-value">{power_val:.1f}%</div><div class="metric-label">Statistical Power</div></div>', unsafe_allow_html=True)
        else:
            st.markdown("### **Sample Size**")
            
            # Create sample size dataframe
            if study_design == "Two independent study groups":
                # Show both unadjusted and adjusted sample sizes if dropout applied
                if dropout_rate > 0:
                    sample_df = pd.DataFrame({
                        "Group": ["Group 1", "Group 2", "**Total**", "", "Without dropout:", "Group 1 (base)", "Group 2 (base)", "**Total (base)**"],
                        "Size": [results['n1'], results['n2'], f"**{results['total']}**", "", "", 
                                results.get('n1_unadjusted', 'N/A'), 
                                results.get('n2_unadjusted', 'N/A'), 
                                f"**{results.get('total_unadjusted', 'N/A')}**"]
                    })
                else:
                    sample_df = pd.DataFrame({
                        "Group": ["Group 1", "Group 2", "**Total**"],
                        "Size": [results['n1'], results['n2'], f"**{results['total']}**"]
                    })
                st.markdown(f'<div class="metric-card"><div class="metric-value">{results["total"]}</div><div class="metric-label">Total Sample Size</div></div>', unsafe_allow_html=True)
            else:
                # Show both unadjusted and adjusted for one-group studies
                if dropout_rate > 0:
                    sample_df = pd.DataFrame({
                        "Group": ["Study Group", "**Total**", "", "Without dropout:", "Study Group (base)", "**Total (base)**"],
                        "Size": [results['n'], f"**{results['n']}**", "", "", 
                                results.get('n_unadjusted', 'N/A'), 
                                f"**{results.get('n_unadjusted', 'N/A')}**"]
                    })
                else:
                    sample_df = pd.DataFrame({
                        "Group": ["Study Group", "**Total**"],
                        "Size": [results['n'], f"**{results['n']}**"]
                    })
                st.markdown(f'<div class="metric-card"><div class="metric-value">{results["n"]}</div><div class="metric-label">Required Sample Size</div></div>', unsafe_allow_html=True)
            
            st.dataframe(
                sample_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Group": st.column_config.TextColumn("Group", width="medium"),
                    "Size": st.column_config.TextColumn("Size", width="medium")
                }
            )
    
    with col2:
        st.markdown("### **Study Parameters**")
        
        # Build parameters dataframe based on context
        if study_design == "Two independent study groups":
            if outcome_type == "Continuous (means)":
                if is_posthoc:
                    param_df = pd.DataFrame({
                        "Parameter": ["Sample size, group 1", "Sample size, group 2", "Mean, group 1", "Mean, group 2", "Standard deviation", "Alpha", "Power"],
                        "Value": [params['n1'], params['n2'], params['mean1'], params['mean2'], params['std_dev'], 
                                 params['alpha'], f"{power_val:.1f}%"]
                    })
                else:
                    std_display = params.get('pooled_std', params.get('std_dev', 'N/A'))
                    param_data = [
                        ["Mean, group 1", params['mean1']], 
                        ["Mean, group 2", params['mean2']], 
                        ["Pooled standard deviation", f"{std_display:.2f}"],
                        ["Alpha", params['alpha']], 
                        ["Beta", round(1-params['power'], 2)], 
                        ["Power", params['power']]
                    ]
                    if dropout_rate > 0:
                        param_data.append(["Dropout rate", f"{dropout_rate*100:.0f}%"])
                    
                    param_df = pd.DataFrame({
                        "Parameter": [item[0] for item in param_data],
                        "Value": [item[1] for item in param_data]
                    })
            else:
                # Similar implementation for other cases...
                param_df = pd.DataFrame({
                    "Parameter": ["Proportion, group 1", "Proportion, group 2", "Alpha", "Beta", "Power"],
                    "Value": [f"{params.get('p1', 0):.2f}", f"{params.get('p2', 0):.2f}", 
                             params.get('alpha', 0.05), round(1-params.get('power', 0.8), 2), params.get('power', 0.8)]
                })
        else:
            # One group implementations...
            param_df = pd.DataFrame({
                "Parameter": ["Sample mean", "Population mean", "Standard deviation", "Alpha", "Power"],
                "Value": [params.get('sample_mean', 0), params.get('population_mean', 0), 
                         params.get('std_dev', 0), params.get('alpha', 0.05), params.get('power', 0.8)]
            })
        
        # Display parameters table
        st.dataframe(
            param_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Parameter": st.column_config.TextColumn("Parameter", width="large"),
                "Value": st.column_config.TextColumn("Value", width="medium")
            }
        )
        
        # Effect size display
        st.metric("Effect Size", f"{results['effect_size']:.4f}")

def create_posthoc_visualizations(study_design, outcome_type, params, results):
    """Create post-hoc power analysis visualizations"""
    
    st.markdown('<div class="visualization-container">', unsafe_allow_html=True)
    st.markdown("### 📊 **Post-Hoc Power Analysis Visualizations**")
    
    # Create tabs for different visualization types
    tab1, tab2, tab3 = st.tabs(["📈 Power Curves", "🎯 Sample Size Impact", "📊 Effect Size Analysis"])
    
    with tab1:
        # Power vs Effect Size curves
        if study_design == "Two independent study groups":
            if outcome_type == "Dichotomous (yes/no)":
                # Power vs effect size for two proportions
                p1_base = params.get('p1', 0.2)
                effect_sizes = np.linspace(0.05, 0.5, 30)
                powers = []
                
                n1 = params.get('n1', 50)
                n2 = params.get('n2', 50)
                alpha = params.get('alpha', 0.05)
                
                for es in effect_sizes:
                    p2_test = p1_base + es
                    if p2_test <= 0.99:
                        try:
                            power_result = PostHocPowerAnalyzer.calculate_power_two_proportions(
                                n1, n2, p1_base, p2_test, alpha, True
                            )
                            powers.append(power_result['power'] * 100)
                        except:
                            powers.append(0)
                    else:
                        powers.append(np.nan)
                
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(
                    x=effect_sizes, y=powers,
                    mode='lines+markers',
                    name='Power vs Effect Size',
                    line=dict(color='#2E86AB', width=4),
                    marker=dict(size=8)
                ))
                
                # Add marker for current study
                current_effect = results.get('effect_size', 0)
                current_power = results.get('power_percent', 0)
                
                fig1.add_trace(go.Scatter(
                    x=[current_effect],
                    y=[current_power],
                    mode='markers',
                    name='Your Study',
                    marker=dict(size=15, color='red', symbol='star',
                               line=dict(width=2, color='white'))
                ))
                
                fig1.add_hline(y=80, line_dash="dash", line_color="green", 
                               annotation_text="80% Power Threshold")
                
                fig1.update_layout(
                    title="Statistical Power vs Effect Size",
                    xaxis_title="Effect Size (Absolute Difference in Proportions)",
                    yaxis_title="Statistical Power (%)",
                    height=500,
                    yaxis=dict(range=[0, 100])
                )
                
                st.plotly_chart(fig1, use_container_width=True)
                
            elif outcome_type == "Continuous (means)":
                # Power vs effect size for continuous outcomes
                mean1_base = params.get('mean1', 10)
                std_dev = params.get('std_dev', 2)
                effect_sizes = np.linspace(0.2, 2.0, 30)
                powers = []
                
                n1 = params.get('n1', 25)
                n2 = params.get('n2', 25)
                alpha = params.get('alpha', 0.05)
                
                for es in effect_sizes:
                    mean2_test = mean1_base + es * std_dev
                    try:
                        power_result = PostHocPowerAnalyzer.calculate_power_two_means(
                            n1, n2, mean1_base, mean2_test, std_dev, alpha, True
                        )
                        powers.append(power_result['power'] * 100)
                    except:
                        powers.append(0)
                
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(
                    x=effect_sizes, y=powers,
                    mode='lines+markers',
                    name='Power vs Effect Size',
                    line=dict(color='#2E86AB', width=4),
                    marker=dict(size=8)
                ))
                
                # Add marker for current study
                current_effect = results.get('effect_size', 0)
                current_power = results.get('power_percent', 0)
                
                fig1.add_trace(go.Scatter(
                    x=[current_effect],
                    y=[current_power],
                    mode='markers',
                    name='Your Study',
                    marker=dict(size=15, color='red', symbol='star',
                               line=dict(width=2, color='white'))
                ))
                
                fig1.add_hline(y=80, line_dash="dash", line_color="green", 
                               annotation_text="80% Power Threshold")
                
                fig1.update_layout(
                    title="Statistical Power vs Effect Size (Cohen's d)",
                    xaxis_title="Effect Size (Cohen's d)",
                    yaxis_title="Statistical Power (%)",
                    height=500,
                    yaxis=dict(range=[0, 100])
                )
                
                st.plotly_chart(fig1, use_container_width=True)
        
        else:  # One group vs population post-hoc
            if outcome_type == "Dichotomous (yes/no)":
                # Power vs effect size for one proportion
                p0_base = params.get('population_prop', 0.25)
                effect_sizes = np.linspace(0.05, 0.4, 30)
                powers = []
                
                n = params.get('n', 60)
                alpha = params.get('alpha', 0.05)
                
                for es in effect_sizes:
                    p1_test = p0_base + es
                    if p1_test <= 0.99:
                        try:
                            power_result = PostHocPowerAnalyzer.calculate_power_one_proportion(
                                n, p1_test, p0_base, alpha, True
                            )
                            powers.append(power_result['power'] * 100)
                        except:
                            powers.append(0)
                    else:
                        powers.append(np.nan)
                
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(
                    x=effect_sizes, y=powers,
                    mode='lines+markers',
                    name='Power vs Effect Size',
                    line=dict(color='#2E86AB', width=4),
                    marker=dict(size=8)
                ))
                
                # Add marker for current study
                current_effect = results.get('effect_size', 0)
                current_power = results.get('power_percent', 0)
                
                fig1.add_trace(go.Scatter(
                    x=[current_effect],
                    y=[current_power],
                    mode='markers',
                    name='Your Study',
                    marker=dict(size=15, color='red', symbol='star',
                               line=dict(width=2, color='white'))
                ))
                
                fig1.add_hline(y=80, line_dash="dash", line_color="green", 
                               annotation_text="80% Power Threshold")
                
                fig1.update_layout(
                    title="Statistical Power vs Effect Size",
                    xaxis_title="Effect Size (Absolute Difference in Proportions)",
                    yaxis_title="Statistical Power (%)",
                    height=500,
                    yaxis=dict(range=[0, 100])
                )
                
                st.plotly_chart(fig1, use_container_width=True)
                
            elif outcome_type == "Continuous (means)":
                # Power vs effect size for one-sample continuous
                mean_pop = params.get('population_mean', 10)
                std_dev = params.get('std_dev', 2)
                effect_sizes = np.linspace(0.2, 2.0, 30)
                powers = []
                
                n = params.get('n', 30)
                alpha = params.get('alpha', 0.05)
                
                for es in effect_sizes:
                    mean_sample_test = mean_pop + es * std_dev
                    try:
                        power_result = PostHocPowerAnalyzer.calculate_power_one_mean(
                            n, mean_sample_test, mean_pop, std_dev, alpha, True
                        )
                        powers.append(power_result['power'] * 100)
                    except:
                        powers.append(0)
                
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(
                    x=effect_sizes, y=powers,
                    mode='lines+markers',
                    name='Power vs Effect Size',
                    line=dict(color='#2E86AB', width=4),
                    marker=dict(size=8)
                ))
                
                # Add marker for current study
                current_effect = results.get('effect_size', 0)
                current_power = results.get('power_percent', 0)
                
                fig1.add_trace(go.Scatter(
                    x=[current_effect],
                    y=[current_power],
                    mode='markers',
                    name='Your Study',
                    marker=dict(size=15, color='red', symbol='star',
                               line=dict(width=2, color='white'))
                ))
                
                fig1.add_hline(y=80, line_dash="dash", line_color="green", 
                               annotation_text="80% Power Threshold")
                
                fig1.update_layout(
                    title="Statistical Power vs Effect Size (Cohen's d)",
                    xaxis_title="Effect Size (Cohen's d)",
                    yaxis_title="Statistical Power (%)",
                    height=500,
                    yaxis=dict(range=[0, 100])
                )
                
                st.plotly_chart(fig1, use_container_width=True)
    
    with tab2:
        # Sample size impact analysis
        st.markdown("#### **Sample Size Impact on Power**")
        
        if study_design == "Two independent study groups":
            sample_sizes = np.arange(10, 200, 5)
            powers = []
            
            if outcome_type == "Dichotomous (yes/no)":
                p1 = params.get('p1', 0.2)
                p2 = params.get('p2', 0.3)
                alpha = params.get('alpha', 0.05)
                
                for n in sample_sizes:
                    try:
                        power_result = PostHocPowerAnalyzer.calculate_power_two_proportions(
                            n, n, p1, p2, alpha, True
                        )
                        powers.append(power_result['power'] * 100)
                    except:
                        powers.append(0)
                        
            else:  # Continuous
                mean1 = params.get('mean1', 10)
                mean2 = params.get('mean2', 12)
                std_dev = params.get('std_dev', 2)
                alpha = params.get('alpha', 0.05)
                
                for n in sample_sizes:
                    try:
                        power_result = PostHocPowerAnalyzer.calculate_power_two_means(
                            n, n, mean1, mean2, std_dev, alpha, True
                        )
                        powers.append(power_result['power'] * 100)
                    except:
                        powers.append(0)
            
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=sample_sizes * 2,  # Total sample size
                y=powers,
                mode='lines+markers',
                name='Power vs Sample Size',
                line=dict(color='#A23B72', width=4),
                marker=dict(size=6)
            ))
            
            # Add marker for current study
            current_n1 = params.get('n1', 25)
            current_n2 = params.get('n2', 25)
            current_total = current_n1 + current_n2
            current_power = results.get('power_percent', 0)
            
            fig2.add_trace(go.Scatter(
                x=[current_total],
                y=[current_power],
                mode='markers',
                name='Your Study',
                marker=dict(size=15, color='red', symbol='star',
                           line=dict(width=2, color='white'))
            ))
            
            fig2.add_hline(y=80, line_dash="dash", line_color="green", 
                           annotation_text="80% Power Threshold")
            
            fig2.update_layout(
                title="Statistical Power vs Total Sample Size",
                xaxis_title="Total Sample Size",
                yaxis_title="Statistical Power (%)",
                height=500,
                yaxis=dict(range=[0, 100])
            )
            
            st.plotly_chart(fig2, use_container_width=True)
        
        else:  # One group vs population
            sample_sizes = np.arange(10, 200, 5)
            powers = []
            
            if outcome_type == "Dichotomous (yes/no)":
                sample_prop = params.get('sample_prop', 0.18)
                pop_prop = params.get('population_prop', 0.25)
                alpha = params.get('alpha', 0.05)
                
                for n in sample_sizes:
                    try:
                        power_result = PostHocPowerAnalyzer.calculate_power_one_proportion(
                            n, sample_prop, pop_prop, alpha, True
                        )
                        powers.append(power_result['power'] * 100)
                    except:
                        powers.append(0)
                        
            else:  # Continuous
                sample_mean = params.get('sample_mean', 11.5)
                pop_mean = params.get('population_mean', 10)
                std_dev = params.get('std_dev', 2)
                alpha = params.get('alpha', 0.05)
                
                for n in sample_sizes:
                    try:
                        power_result = PostHocPowerAnalyzer.calculate_power_one_mean(
                            n, sample_mean, pop_mean, std_dev, alpha, True
                        )
                        powers.append(power_result['power'] * 100)
                    except:
                        powers.append(0)
            
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=sample_sizes,
                y=powers,
                mode='lines+markers',
                name='Power vs Sample Size',
                line=dict(color='#A23B72', width=4),
                marker=dict(size=6)
            ))
            
            # Add marker for current study
            current_n = params.get('n', 60)
            current_power = results.get('power_percent', 0)
            
            fig2.add_trace(go.Scatter(
                x=[current_n],
                y=[current_power],
                mode='markers',
                name='Your Study',
                marker=dict(size=15, color='red', symbol='star',
                           line=dict(width=2, color='white'))
            ))
            
            fig2.add_hline(y=80, line_dash="dash", line_color="green", 
                           annotation_text="80% Power Threshold")
            
            fig2.update_layout(
                title="Statistical Power vs Sample Size",
                xaxis_title="Sample Size",
                yaxis_title="Statistical Power (%)",
                height=500,
                yaxis=dict(range=[0, 100])
            )
            
            st.plotly_chart(fig2, use_container_width=True)
    
    with tab3:
        # Effect size interpretation and recommendations
        st.markdown("#### **Effect Size Analysis & Recommendations**")
        current_effect = results.get('effect_size', 0)
        current_power = results.get('power_percent', 0)
        
        if outcome_type == "Continuous (means)":
            if current_effect < 0.2:
                effect_interpretation = "Small"
                color = "info"
            elif current_effect < 0.5:
                effect_interpretation = "Medium"  
                color = "success"
            else:
                effect_interpretation = "Large"
                color = "success"
                
            if color == "info":
                st.info(f"**{effect_interpretation} Effect Size** (Cohen's d = {current_effect:.3f}): Small effects are harder to detect and require larger sample sizes.")
            else:
                st.success(f"**{effect_interpretation} Effect Size** (Cohen's d = {current_effect:.3f}): This effect size is considered {effect_interpretation.lower()} in magnitude.")
        else:
            st.info(f"**Proportion Difference** = {current_effect:.3f}: Absolute difference between group proportions.")
        
        # Power interpretation
        st.markdown("#### **Power Analysis Results**")
        if current_power >= 80:
            st.success(f"**Adequate Power ({current_power:.1f}%):** Your study had sufficient power to detect the observed effect size.")
        elif current_power >= 50:
            st.warning(f"**Moderate Power ({current_power:.1f}%):** Your study had moderate power. Consider larger sample sizes for future studies.")
        else:
            st.error(f"**Low Power ({current_power:.1f}%):** Your study was underpowered to detect the observed effect size.")
        
        # Recommendations for future studies
        st.markdown("#### **Recommendations for Future Studies**")
        if current_power < 80:
            # Calculate required sample size for 80% power
            if study_design == "Two independent study groups":
                if outcome_type == "Dichotomous (yes/no)":
                    req_result = SampleSizeCalculator.calculate_proportions_two_groups(
                        params.get('p1', 0.2),
                        params.get('p2', 0.3),
                        params.get('alpha', 0.05),
                        0.80
                    )
                    req_sample = req_result['total']
                    current_sample = params.get('n1', 25) + params.get('n2', 25)
                else:
                    req_result = SampleSizeCalculator.calculate_continuous_two_groups(
                        params.get('mean1', 10),
                        params.get('mean2', 12),
                        params.get('std_dev', 2),
                        params.get('std_dev', 2),
                        params.get('alpha', 0.05),
                        0.80
                    )
                    req_sample = req_result['total']
                    current_sample = params.get('n1', 25) + params.get('n2', 25)
                    
                st.markdown(f"""
                **To achieve 80% power with the same effect size:**
                - Required total sample size: **{req_sample}** subjects
                - Your study had: **{current_sample}** subjects  
                - Additional subjects needed: **{max(0, req_sample - current_sample)}**
                """)
            else:  # One group studies
                if outcome_type == "Dichotomous (yes/no)":
                    req_result = SampleSizeCalculator.calculate_proportions_one_group(
                        params.get('sample_prop', 0.18),
                        params.get('population_prop', 0.25),
                        params.get('alpha', 0.05),
                        0.80
                    )
                    req_sample = req_result['n']
                    current_sample = params.get('n', 60)
                else:
                    req_result = SampleSizeCalculator.calculate_continuous_one_group(
                        params.get('sample_mean', 11.5),
                        params.get('population_mean', 10),
                        params.get('std_dev', 2),
                        params.get('alpha', 0.05),
                        0.80
                    )
                    req_sample = req_result['n']
                    current_sample = params.get('n', 30)
                
                st.markdown(f"""
                **To achieve 80% power with the same effect size:**
                - Required sample size: **{req_sample}** subjects
                - Your study had: **{current_sample}** subjects
                - Additional subjects needed: **{max(0, req_sample - current_sample)}**
                """)
    
    st.markdown('</div>', unsafe_allow_html=True)

def generate_citations(study_design, outcome_type, params, results):
    """Generate multiple citation formats"""
    
    current_date = datetime.now()
    date_accessed = current_date.strftime("%B %d, %Y")
    date_mla = current_date.strftime("%d %b %Y")
    
    # Determine study details
    if study_design == "Two independent study groups":
        study_desc = "two independent study groups"
    else:
        study_desc = "one study group vs. population"
    
    if outcome_type == "Continuous (means)":
        outcome_desc = "continuous"
    else:
        outcome_desc = "dichotomous (yes/no)"
    
    # Sample size info
    if study_design == "Two independent study groups":
        size_info = f"n₁={results['n1']}, n₂={results['n2']}, total={results['total']}"
    else:
        size_info = f"N={results['n']}"
    
    # Parameter string
    param_str = f"α={params.get('alpha', 0.05)}, β={1-params.get('power', 0.80):.1f}, power={params.get('power', 0.80)*100:.0f}%"
    
    citations = {
        "APA": f"ClinCalc Sample Size Calculator. ({current_date.year}). Sample size calculation for {study_desc}, {outcome_desc} outcome. "
               f"Statistical parameters: {param_str}. Results: {size_info}. Retrieved {date_accessed}, from https://github.com/ArioMoniri/samplecalc",
        
        "MLA": f"Sample Size Calculator. ClinCalc, {current_date.year}, github.com/ArioMoniri/samplecalc. Accessed {date_mla}. "
               f"Study design: {study_desc}, {outcome_desc}. Parameters: {param_str}. Sample size: {size_info}.",
        
        "Chicago": f"ClinCalc Sample Size Calculator. \"Sample Size Calculation Results.\" GitHub. Accessed {date_accessed}. "
                   f"https://github.com/ArioMoniri/samplecalc. Study: {study_desc}, {outcome_desc}. {param_str}. Required sample size: {size_info}.",
        
        "Vancouver": f"ClinCalc Sample Size Calculator [Internet]. Sample size calculation for {study_desc} study with {outcome_desc} outcome. "
                     f"{param_str}. Sample size required: {size_info}. [cited {current_date.year} {current_date.strftime('%b %d')}]. "
                     f"Available from: https://github.com/ArioMoniri/samplecalc",
        
        "Harvard": f"ClinCalc Sample Size Calculator ({current_date.year}) Sample size calculation results. Available at: "
                   f"https://github.com/ArioMoniri/samplecalc (Accessed: {date_accessed}). Study design: {study_desc}, outcome: {outcome_desc}. "
                   f"Statistical parameters: {param_str}. Required sample size: {size_info}."
    }
    
    return citations

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🧮 ClinCalc Sample Size Calculator</h1>
        <p>Professional Statistical Power Analysis for Clinical Research</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Unified sidebar configuration for both tabs
    with st.sidebar:
        st.markdown("## 📊 Study Configuration")
        
        st.markdown("**Select Study Design:**")
        study_design = st.radio(
            "",
            ["👥 Two independent study groups", "👤 One study group vs. population"],
            help="Choose your study design type"
        )
        
        # Clean up design names
        if "Two independent" in study_design:
            study_design = "Two independent study groups"
        else:
            study_design = "One study group vs. population"
        
        st.markdown("**Select Outcome Type:**")
        outcome_type = st.radio(
            "",
            ["🔘 Dichotomous (yes/no)", "📊 Continuous (means)"],
            help="Select the type of your primary outcome variable"
        )
        
        # Clean up outcome names
        if "Dichotomous" in outcome_type:
            outcome_type = "Dichotomous (yes/no)"
        else:
            outcome_type = "Continuous (means)"
        
        st.markdown("---")
        st.markdown("### ⚙️ Statistical Parameters")
        
        confidence_level = st.selectbox(
            "**Confidence Level (%)**",
            [90, 95, 99],
            index=1,
            help="The confidence level determines the probability that the confidence interval contains the true population parameter"
        )
        alpha = (100 - confidence_level) / 100
        
        power_percent = st.selectbox(
            "**Statistical Power (%)**",
            [70, 80, 90, 95],
            index=1,
            help="Statistical power is the probability of correctly rejecting a false null hypothesis (avoiding Type II error)"
        )
        power = power_percent / 100
        
        two_sided = st.checkbox(
            "**Two-sided test**",
            value=True,
            help="Two-sided tests detect differences in either direction, while one-sided tests only detect differences in one specified direction"
        )
        
        dropout_rate = create_synchronized_input(
            "Expected dropout rate (%)",
            0.0, 50.0, 10.0, 1.0, 
            "Percentage of participants expected to drop out or be lost to follow-up during the study",
            "dropout"
        ) / 100
    
    # Main navigation tabs
    main_tab1, main_tab2, main_tab3 = st.tabs(["📊 Sample Size Analysis", "🔄 Post-Hoc Power Analysis", "📚 Education & Resources"])
    
    with main_tab1:
        
        # Main content area with study type info
        display_study_type_info(study_design, outcome_type)
        
        # Sub-tabs
        sub_tab1, sub_tab2, sub_tab3, sub_tab4 = st.tabs(["📋 Calculator", "🧮 Formula", "📊 Analysis", "📄 Citation"])
        
        with sub_tab1:
            st.markdown('<div class="calc-container">', unsafe_allow_html=True)
            
            # Dynamic form based on selections
            if study_design == "Two independent study groups":
                if outcome_type == "Continuous (means)":
                    st.markdown("### **Means:**")
                    
                    # Mean specification type
                    mean_type = st.selectbox(
                        "**Specify means as:**",
                        ["Mean ± Standard Deviation", "% Increase", "% Decrease"],
                        help="Choose how to specify the group means"
                    )
                    
                    if mean_type == "Mean ± Standard Deviation":
                        col_a, col_b = st.columns(2)
                        with col_a:
                            mean1 = create_synchronized_input("Group 1 mean", -1000.0, 1000.0, 10.0, 0.1, 
                                                            "Expected mean for group 1", "mean1", "%.1f")
                        with col_b:
                            mean2 = create_synchronized_input("Group 2 mean", -1000.0, 1000.0, 12.0, 0.1,
                                                            "Expected mean for group 2", "mean2", "%.1f")
                                                            
                    elif mean_type == "% Increase":
                        # Direct percentage inputs for each group
                        col_a, col_b = st.columns(2)
                        with col_a:
                            mean1_pct = create_synchronized_input("Group 1 % change", -99.0, 500.0, 0.0, 1.0,
                                                                "Percentage change for group 1", "mean1_pct", "%.0f")
                        with col_b:
                            mean2_pct = create_synchronized_input("Group 2 % change", -99.0, 500.0, 20.0, 1.0,
                                                                "Percentage change for group 2", "mean2_pct", "%.0f")
                        
                        baseline_mean = create_synchronized_input("Baseline value", 0.1, 1000.0, 10.0, 0.1,
                                                                "Baseline value for percentage calculations", "baseline_pct", "%.1f")
                        mean1 = baseline_mean * (1 + mean1_pct/100)
                        mean2 = baseline_mean * (1 + mean2_pct/100)
                        st.write(f"Group 1: {mean1:.1f} ± SD₁, Group 2: {mean2:.1f} ± SD₂")
                        
                    elif mean_type == "% Decrease":
                        col_a, col_b = st.columns(2)
                        with col_a:
                            mean1_pct = create_synchronized_input("Group 1 % decrease", 1.0, 99.0, 16.7, 1.0,
                                                                "Percentage decrease for group 1", "mean1_dec", "%.0f")
                        with col_b:
                            mean2_pct = create_synchronized_input("Group 2 % decrease", 1.0, 99.0, 0.0, 1.0,
                                                                "Percentage decrease for group 2", "mean2_dec", "%.0f")
                        
                        baseline_mean = create_synchronized_input("Baseline value", 0.1, 1000.0, 12.0, 0.1,
                                                                "Baseline value for percentage calculations", "baseline_dec", "%.1f")
                        mean1 = baseline_mean * (1 - mean1_pct/100)
                        mean2 = baseline_mean * (1 - mean2_pct/100)
                        st.write(f"Group 1: {mean1:.1f} ± SD₁, Group 2: {mean2:.1f} ± SD₂")
                    
                    # Separate standard deviations for each group
                    col_std1, col_std2 = st.columns(2)
                    with col_std1:
                        std_dev1 = create_synchronized_input("Group 1 standard deviation", 0.01, 100.0, 2.0, 0.1,
                                                           "Standard deviation for group 1", "std_dev1", "%.1f")
                    with col_std2:
                        std_dev2 = create_synchronized_input("Group 2 standard deviation", 0.01, 100.0, 2.0, 0.1,
                                                           "Standard deviation for group 2", "std_dev2", "%.1f")
                    
                    allocation_ratio = create_synchronized_input("Allocation ratio (group 2 / group 1)", 0.1, 5.0, 1.0, 0.1,
                                                               "Ratio of group 2 size to group 1 size", "allocation", "%.1f")
                    
                    if st.button("🔢 **Calculate Sample Size**", type="primary", use_container_width=True, key="calc_continuous_two"):
                        try:
                            results = SampleSizeCalculator.calculate_continuous_two_groups(
                                mean1, mean2, std_dev1, std_dev2, alpha, power, allocation_ratio, two_sided, dropout_rate
                            )
                            
                            params = {
                                'mean1': mean1, 'mean2': mean2, 
                                'std_dev1': std_dev1, 'std_dev2': std_dev2,
                                'pooled_std': results['pooled_std'],
                                'alpha': alpha, 'power': power, 'z_alpha': results['z_alpha'],
                                'z_beta': results['z_beta'], 'allocation_ratio': allocation_ratio,
                                'dropout_rate': dropout_rate
                            }
                            
                            st.session_state.results = results
                            st.session_state.params = params
                            st.session_state.study_design = study_design
                            st.session_state.outcome_type = outcome_type
                            
                            display_professional_results_tables(results, study_design, outcome_type, params)
                            
                        except Exception as e:
                            st.error(f"Calculation error: {str(e)}")
                
                else:  # Dichotomous outcomes
                    st.markdown("### **Proportions:**")
                    
                    # Enrollment ratio options
                    ratio_type = st.selectbox(
                        "**Enrollment Ratio Type**",
                        ["Incidence", "% Increase", "% Decrease"],
                        help="Choose how to specify the difference between groups"
                    )
                    
                    if ratio_type == "Incidence":
                        p1 = create_synchronized_input("Group 1 proportion", 0.01, 0.99, 0.14, 0.01, 
                                                     "Expected proportion in group 1", "p1", "%.2f")
                        p2 = create_synchronized_input("Group 2 proportion", 0.01, 0.99, 0.21, 0.01,
                                                     "Expected proportion in group 2", "p2", "%.2f")
                                                     
                    elif ratio_type == "% Increase":
                        # Direct percentage inputs for each group
                        col_a, col_b = st.columns(2)
                        with col_a:
                            p1_pct = create_synchronized_input("Group 1 % increase", 0.0, 500.0, 0.0, 1.0,
                                                             "Percentage increase for group 1", "p1_inc", "%.0f")
                        with col_b:
                            p2_pct = create_synchronized_input("Group 2 % increase", 0.0, 500.0, 50.0, 1.0,
                                                             "Percentage increase for group 2", "p2_inc", "%.0f")
                        
                        baseline_prop = create_synchronized_input("Baseline proportion", 0.01, 0.99, 0.14, 0.01,
                                                                "Baseline proportion for calculations", "baseline_prop_inc", "%.2f")
                        p1 = min(0.99, baseline_prop * (1 + p1_pct/100))
                        p2 = min(0.99, baseline_prop * (1 + p2_pct/100))
                        st.write(f"Group 1: {p1:.3f}, Group 2: {p2:.3f}")
                        
                    elif ratio_type == "% Decrease":
                        col_a, col_b = st.columns(2)
                        with col_a:
                            p1_pct = create_synchronized_input("Group 1 % decrease", 1.0, 99.0, 33.0, 1.0,
                                                             "Percentage decrease for group 1", "p1_dec", "%.0f")
                        with col_b:
                            p2_pct = create_synchronized_input("Group 2 % decrease", 1.0, 99.0, 0.0, 1.0,
                                                             "Percentage decrease for group 2", "p2_dec", "%.0f")
                        
                        baseline_prop = create_synchronized_input("Baseline proportion", 0.01, 0.99, 0.21, 0.01,
                                                                "Baseline proportion for calculations", "baseline_prop_dec", "%.2f")
                        p1 = max(0.01, baseline_prop * (1 - p1_pct/100))
                        p2 = max(0.01, baseline_prop * (1 - p2_pct/100))
                        st.write(f"Group 1: {p1:.3f}, Group 2: {p2:.3f}")
                    
                    allocation_ratio = create_synchronized_input("Allocation ratio (group 2 / group 1)", 0.1, 5.0, 1.0, 0.1,
                                                               "Ratio of group 2 size to group 1 size", "allocation_prop", "%.1f")
                    
                    if st.button("🔢 **Calculate Sample Size**", type="primary", use_container_width=True, key="calc_dichotomous_two"):
                        try:
                            results = SampleSizeCalculator.calculate_proportions_two_groups(
                                p1, p2, alpha, power, allocation_ratio, two_sided, True, dropout_rate
                            )
                            
                            params = {
                                'p1': p1, 'p2': p2, 'alpha': alpha, 'power': power,
                                'z_alpha': results['z_alpha'], 'z_beta': results['z_beta'],
                                'p_pooled': results['p_pooled'], 'allocation_ratio': allocation_ratio,
                                'dropout_rate': dropout_rate
                            }
                            
                            st.session_state.results = results
                            st.session_state.params = params
                            st.session_state.study_design = study_design
                            st.session_state.outcome_type = outcome_type
                            
                            display_professional_results_tables(results, study_design, outcome_type, params)
                            
                        except Exception as e:
                            st.error(f"Calculation error: {str(e)}")
            
            else:  # One group vs population implementations...
                if outcome_type == "Continuous (means)":
                    st.markdown("### **Means:**")
                    
                    # Mean specification type
                    mean_type = st.selectbox(
                        "**Specify means as:**",
                        ["Mean ± Standard Deviation", "% Increase", "% Decrease"],
                        help="Choose how to specify the expected difference"
                    )
                    
                    if mean_type == "Mean ± Standard Deviation":
                        col_a, col_b = st.columns(2)
                        with col_a:
                            sample_mean = create_synchronized_input("Expected sample mean", -1000.0, 1000.0, 12.0, 0.1,
                                                                  "Expected mean in your study", "sample_mean", "%.1f")
                        with col_b:
                            population_mean = create_synchronized_input("Population mean", -1000.0, 1000.0, 10.0, 0.1,
                                                                      "Known population mean", "pop_mean", "%.1f")
                                                                      
                    elif mean_type == "% Increase":
                        population_mean = create_synchronized_input("Population mean", -1000.0, 1000.0, 10.0, 0.1,
                                                                  "Known population mean", "pop_mean_inc", "%.1f")
                        sample_increase_pct = create_synchronized_input("Expected % increase", 1.0, 500.0, 20.0, 1.0,
                                                                      "Expected percentage increase for study group", "sample_increase", "%.0f")
                        sample_mean = population_mean * (1 + sample_increase_pct/100)
                        st.write(f"Population: {population_mean:.1f} ± SD, Expected Study: {sample_mean:.1f} ± SD")
                        
                    elif mean_type == "% Decrease":
                        population_mean = create_synchronized_input("Population mean", -1000.0, 1000.0, 12.0, 0.1,
                                                                  "Known population mean", "pop_mean_dec", "%.1f")
                        sample_decrease_pct = create_synchronized_input("Expected % decrease", 1.0, 99.0, 16.7, 1.0,
                                                                      "Expected percentage decrease for study group", "sample_decrease", "%.0f")
                        sample_mean = population_mean * (1 - sample_decrease_pct/100)
                        st.write(f"Population: {population_mean:.1f} ± SD, Expected Study: {sample_mean:.1f} ± SD")
                    
                    std_dev = create_synchronized_input("Standard deviation", 0.01, 100.0, 2.0, 0.1,
                                                      "Population standard deviation", "std_dev_one", "%.1f")
                    
                    if st.button("🔢 **Calculate Sample Size**", type="primary", use_container_width=True, key="calc_continuous_one"):
                        try:
                            results = SampleSizeCalculator.calculate_continuous_one_group(
                                sample_mean, population_mean, std_dev, alpha, power, two_sided, dropout_rate
                            )
                            
                            params = {
                                'sample_mean': sample_mean, 'population_mean': population_mean,
                                'std_dev': std_dev, 'alpha': alpha, 'power': power,
                                'z_alpha': results['z_alpha'], 'z_beta': results['z_beta'],
                                'effect_size': results['effect_size'], 'dropout_rate': dropout_rate
                            }
                            
                            st.session_state.results = results
                            st.session_state.params = params
                            st.session_state.study_design = study_design
                            st.session_state.outcome_type = outcome_type
                            
                            display_professional_results_tables(results, study_design, outcome_type, params)
                            
                        except Exception as e:
                            st.error(f"Calculation error: {str(e)}")
                
                else:  # Dichotomous outcomes
                    st.markdown("### **Proportions:**")
                    
                    # Add ratio type selector for one group studies
                    ratio_type = st.selectbox(
                        "**Comparison Type**",
                        ["Incidence", "% Increase from Population", "% Decrease from Population"],
                        help="Choose how to specify the difference from population"
                    )
                    
                    if ratio_type == "Incidence":
                        sample_prop = create_synchronized_input("Expected study proportion", 0.01, 0.99, 0.14, 0.01,
                                                              "Expected proportion in your study", "sample_prop", "%.2f")
                        population_prop = create_synchronized_input("Known population proportion", 0.01, 0.99, 0.21, 0.01,
                                                                  "Known population proportion", "pop_prop", "%.2f")
                                                                  
                    elif ratio_type == "% Increase from Population":
                        population_prop = create_synchronized_input("Known population proportion", 0.01, 0.99, 0.21, 0.01,
                                                                  "Known population proportion", "pop_prop_inc", "%.2f")
                        sample_increase_pct = create_synchronized_input("Expected % increase", 1.0, 500.0, 50.0, 1.0,
                                                                      "Expected percentage increase for study group", "sample_inc_one", "%.0f")
                        sample_prop = min(0.99, population_prop * (1 + sample_increase_pct/100))
                        st.write(f"Population: {population_prop:.3f}, Expected Study: {sample_prop:.3f}")
                        
                    elif ratio_type == "% Decrease from Population":
                        population_prop = create_synchronized_input("Known population proportion", 0.01, 0.99, 0.21, 0.01,
                                                                  "Known population proportion", "pop_prop_dec", "%.2f")
                        sample_decrease_pct = create_synchronized_input("Expected % decrease", 1.0, 99.0, 33.0, 1.0,
                                                                      "Expected percentage decrease for study group", "sample_dec_one", "%.0f")
                        sample_prop = max(0.01, population_prop * (1 - sample_decrease_pct/100))
                        st.write(f"Population: {population_prop:.3f}, Expected Study: {sample_prop:.3f}")
                    
                    if st.button("🔢 **Calculate Sample Size**", type="primary", use_container_width=True, key="calc_dichotomous_one"):
                        try:
                            results = SampleSizeCalculator.calculate_proportions_one_group(
                                sample_prop, population_prop, alpha, power, two_sided, True, dropout_rate
                            )
                            
                            params = {
                                'sample_prop': sample_prop, 'population_prop': population_prop,
                                'alpha': alpha, 'power': power, 'z_alpha': results['z_alpha'],
                                'z_beta': results['z_beta'], 'effect_size': results['effect_size'],
                                'p0': results['p0'], 'p1': results['p1'], 'q0': results['q0'], 'q1': results['q1'],
                                'n': results['n'], 'dropout_rate': dropout_rate
                            }
                            
                            st.session_state.results = results
                            st.session_state.params = params
                            st.session_state.study_design = study_design
                            st.session_state.outcome_type = outcome_type
                            
                            display_professional_results_tables(results, study_design, outcome_type, params)
                            
                        except Exception as e:
                            st.error(f"Calculation error: {str(e)}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with sub_tab2:
            if 'params' in st.session_state:
                display_latex_formula_detailed(
                    st.session_state.study_design,
                    st.session_state.outcome_type,
                    st.session_state.params,
                    is_posthoc=False
                )
            else:
                st.info("Please calculate sample size first to view the formula.")
        
        with sub_tab3:
            if 'results' in st.session_state:
                # Create comprehensive visualizations
                create_enhanced_visualizations(
                    st.session_state.study_design,
                    st.session_state.outcome_type,
                    st.session_state.params,
                    st.session_state.results
                )
                
                # Effect size interpretation
                st.markdown("#### **Effect Size Interpretation**")
                effect_size = st.session_state.results.get('effect_size', 0)
                
                if st.session_state.outcome_type == "Continuous (means)":
                    if effect_size < 0.2:
                        st.info(f"**Small Effect Size** (Cohen's d = {effect_size:.3f}): The difference between groups is small.")
                    elif effect_size < 0.5:
                        st.success(f"**Medium Effect Size** (Cohen's d = {effect_size:.3f}): The difference between groups is moderate.")
                    else:
                        st.success(f"**Large Effect Size** (Cohen's d = {effect_size:.3f}): The difference between groups is large.")
                else:
                    st.info(f"**Proportion Difference** = {effect_size:.3f}: Absolute difference between group proportions.")
                
                # Sample size recommendations
                st.markdown("#### **Sample Size Recommendations**")
                current_n = st.session_state.results.get('total', st.session_state.results.get('n', 0))
                
                st.markdown(f"""
                **Your calculated sample size:** {current_n}
                
                **Practical Considerations:**
                - Add 10-20% for potential dropouts if not already included
                - Consider feasibility of recruitment within your timeframe
                - Budget constraints and resource availability
                - Regulatory requirements for your study type
                
                **Alternative Power Levels:**
                - For 70% power: ~{int(current_n * 0.7)} subjects
                - For 90% power: ~{int(current_n * 1.3)} subjects
                - For 95% power: ~{int(current_n * 1.6)} subjects
                """)
                
                # Clinical significance guidance
                st.markdown("#### **Clinical Significance vs Statistical Significance**")
                st.markdown("""
                <div class="info-box">
                <strong>Remember:</strong> Statistical significance doesn't always equal clinical significance. Consider:
                <ul>
                <li>Is the detected difference clinically meaningful?</li>
                <li>What is the minimum clinically important difference (MCID)?</li>
                <li>Cost-effectiveness of the intervention</li>
                <li>Patient quality of life improvements</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
                
            else:
                st.info("Please calculate sample size first to view the analysis.")
        
        with sub_tab4:
            if 'results' in st.session_state:
                st.markdown("### 📄 **Citation Formats**")
                
                citations = generate_citations(
                    st.session_state.study_design,
                    st.session_state.outcome_type,
                    st.session_state.params,
                    st.session_state.results
                )
                
                # Citation format selector
                citation_format = st.selectbox(
                    "**Select Citation Format:**",
                    ["APA", "MLA", "Chicago", "Vancouver", "Harvard"]
                )
                
                # Display selected citation
                st.markdown("#### **Generated Citation:**")
                st.text_area(
                    f"{citation_format} Citation:",
                    value=citations[citation_format],
                    height=150,
                    help="Copy this citation for your research"
                )
                
                # Show all formats in expander
                with st.expander("📋 **View All Citation Formats**"):
                    for format_name, citation in citations.items():
                        st.markdown(f"**{format_name}:**")
                        st.text(citation)
                        st.markdown("---")
            else:
                st.info("Please calculate sample size first to generate citations.")
    
    with main_tab2:
        st.markdown("### 🔄 **Post-Hoc Power Analysis**")
        st.markdown("*Calculate statistical power from completed study results*")
        
        st.markdown("""
        <div class="power-warning">
        <strong>⚠️ Important Note about Post-Hoc Power Analysis:</strong><br>
        Post-hoc power analysis has significant limitations and should be interpreted cautiously. 
        It's typically used to understand why a study may not have detected a significant effect, 
        but it should <strong>not</strong> be used to "explain away" negative results.
        </div>
        """, unsafe_allow_html=True)
        
        # Display study type info for post-hoc using same configuration
        display_study_type_info(study_design, outcome_type)
        
        # Post-hoc sub-tabs
        posthoc_tab1, posthoc_tab2, posthoc_tab3 = st.tabs(["📋 Calculator", "🧮 Formula", "📊 Interpretation"])
        
        with posthoc_tab1:
            st.markdown('<div class="calc-container">', unsafe_allow_html=True)
            
            # Dynamic form for post-hoc analysis using unified configuration
            if study_design == "Two independent study groups":
                if outcome_type == "Continuous (means)":
                    st.markdown("### **Actual Study Results - Continuous Outcomes:**")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Sample Sizes:**")
                        posthoc_n1 = st.number_input("Group 1 sample size", min_value=1, value=25, step=1, key="posthoc_n1")
                        posthoc_n2 = st.number_input("Group 2 sample size", min_value=1, value=25, step=1, key="posthoc_n2")
                        
                    with col2:
                        st.markdown("**Observed Means:**")
                        posthoc_mean1 = create_synchronized_input("Group 1 mean", -1000.0, 1000.0, 10.2, 0.1,
                                                                "Observed mean in group 1", "posthoc_mean1", "%.1f")
                        posthoc_mean2 = create_synchronized_input("Group 2 mean", -1000.0, 1000.0, 11.8, 0.1,
                                                                "Observed mean in group 2", "posthoc_mean2", "%.1f")
                    
                    posthoc_std = create_synchronized_input("Pooled standard deviation", 0.01, 100.0, 2.1, 0.1,
                                                          "Pooled standard deviation from study", "posthoc_std", "%.1f")
                    
                    if st.button("🔍 **Calculate Post-Hoc Power**", type="primary", use_container_width=True, key="posthoc_calc_cont_two"):
                        try:
                            power_results = PostHocPowerAnalyzer.calculate_power_two_means(
                                posthoc_n1, posthoc_n2, posthoc_mean1, posthoc_mean2, posthoc_std, alpha, two_sided
                            )
                            
                            posthoc_params = {
                                'n1': posthoc_n1, 'n2': posthoc_n2, 'mean1': posthoc_mean1, 'mean2': posthoc_mean2,
                                'std_dev': posthoc_std, 'alpha': alpha, 'power': power_results['power']
                            }
                            
                            display_professional_results_tables(power_results, study_design, outcome_type, posthoc_params, is_posthoc=True)
                            
                            st.session_state.posthoc_results = power_results
                            st.session_state.posthoc_params = posthoc_params
                            st.session_state.posthoc_study_design = study_design
                            st.session_state.posthoc_outcome_type = outcome_type
                            
                        except Exception as e:
                            st.error(f"Calculation error: {str(e)}")
                
                else:  # Dichotomous two groups
                    st.markdown("### **Actual Study Results - Dichotomous Outcomes:**")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Sample Sizes:**")
                        posthoc_n1 = st.number_input("Group 1 sample size", min_value=1, value=50, step=1, key="posthoc_n1_prop")
                        posthoc_n2 = st.number_input("Group 2 sample size", min_value=1, value=50, step=1, key="posthoc_n2_prop")
                        
                    with col2:
                        st.markdown("**Observed Proportions:**")
                        posthoc_p1 = create_synchronized_input("Group 1 proportion", 0.01, 0.99, 0.16, 0.01,
                                                             "Observed proportion in group 1", "posthoc_p1", "%.2f")
                        posthoc_p2 = create_synchronized_input("Group 2 proportion", 0.01, 0.99, 0.24, 0.01,
                                                             "Observed proportion in group 2", "posthoc_p2", "%.2f")
                    
                    if st.button("🔍 **Calculate Post-Hoc Power**", type="primary", use_container_width=True, key="posthoc_calc_prop_two"):
                        try:
                            power_results = PostHocPowerAnalyzer.calculate_power_two_proportions(
                                posthoc_n1, posthoc_n2, posthoc_p1, posthoc_p2, alpha, two_sided
                            )
                            
                            posthoc_params = {
                                'n1': posthoc_n1, 'n2': posthoc_n2, 'p1': posthoc_p1, 'p2': posthoc_p2,
                                'alpha': alpha, 'power': power_results['power']
                            }
                            
                            display_professional_results_tables(power_results, study_design, outcome_type, posthoc_params, is_posthoc=True)
                            
                            st.session_state.posthoc_results = power_results
                            st.session_state.posthoc_params = posthoc_params
                            st.session_state.posthoc_study_design = study_design
                            st.session_state.posthoc_outcome_type = outcome_type
                            
                        except Exception as e:
                            st.error(f"Calculation error: {str(e)}")
            
            else:  # One group vs population
                if outcome_type == "Continuous (means)":
                    st.markdown("### **Actual Study Results - Continuous Outcomes:**")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Sample Size:**")
                        posthoc_n = st.number_input("Study group sample size", min_value=1, value=30, step=1, key="posthoc_n_cont")
                        
                    with col2:
                        st.markdown("**Observed Means:**")
                        posthoc_sample_mean = create_synchronized_input("Observed sample mean", -1000.0, 1000.0, 11.5, 0.1,
                                                                      "Observed mean in your study", "posthoc_sample_mean", "%.1f")
                    
                    posthoc_pop_mean = create_synchronized_input("Population mean", -1000.0, 1000.0, 10.0, 0.1,
                                                               "Known population mean", "posthoc_pop_mean", "%.1f")
                    posthoc_std_one = create_synchronized_input("Standard deviation", 0.01, 100.0, 2.0, 0.1,
                                                              "Standard deviation", "posthoc_std_one", "%.1f")
                    
                    if st.button("🔍 **Calculate Post-Hoc Power**", type="primary", use_container_width=True, key="posthoc_calc_cont_one"):
                        try:
                            power_results = PostHocPowerAnalyzer.calculate_power_one_mean(
                                posthoc_n, posthoc_sample_mean, posthoc_pop_mean, posthoc_std_one, alpha, two_sided
                            )
                            
                            posthoc_params = {
                                'n': posthoc_n, 'sample_mean': posthoc_sample_mean, 'population_mean': posthoc_pop_mean,
                                'std_dev': posthoc_std_one, 'alpha': alpha, 'power': power_results['power']
                            }
                            
                            display_professional_results_tables(power_results, study_design, outcome_type, posthoc_params, is_posthoc=True)
                            
                            st.session_state.posthoc_results = power_results
                            st.session_state.posthoc_params = posthoc_params
                            st.session_state.posthoc_study_design = study_design
                            st.session_state.posthoc_outcome_type = outcome_type
                            
                        except Exception as e:
                            st.error(f"Calculation error: {str(e)}")
                
                else:  # Dichotomous one group
                    st.markdown("### **Actual Study Results - Dichotomous Outcomes:**")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Sample Size:**")
                        posthoc_n_prop = st.number_input("Study group sample size", min_value=1, value=60, step=1, key="posthoc_n_prop_one")
                        
                    with col2:
                        st.markdown("**Observed Proportions:**")
                        posthoc_sample_prop = create_synchronized_input("Observed study proportion", 0.01, 0.99, 0.18, 0.01,
                                                                      "Observed proportion in your study", "posthoc_sample_prop", "%.2f")
                    
                    posthoc_pop_prop = create_synchronized_input("Known population proportion", 0.01, 0.99, 0.25, 0.01,
                                                               "Known population proportion", "posthoc_pop_prop", "%.2f")
                    
                    if st.button("🔍 **Calculate Post-Hoc Power**", type="primary", use_container_width=True, key="posthoc_calc_prop_one"):
                        try:
                            power_results = PostHocPowerAnalyzer.calculate_power_one_proportion(
                                posthoc_n_prop, posthoc_sample_prop, posthoc_pop_prop, alpha, two_sided
                            )
                            
                            posthoc_params = {
                                'n': posthoc_n_prop, 'sample_prop': posthoc_sample_prop, 'population_prop': posthoc_pop_prop,
                                'alpha': alpha, 'power': power_results['power']
                            }
                            
                            display_professional_results_tables(power_results, study_design, outcome_type, posthoc_params, is_posthoc=True)
                            
                            st.session_state.posthoc_results = power_results
                            st.session_state.posthoc_params = posthoc_params
                            st.session_state.posthoc_study_design = study_design
                            st.session_state.posthoc_outcome_type = outcome_type
                            
                        except Exception as e:
                            st.error(f"Calculation error: {str(e)}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with posthoc_tab2:
            if 'posthoc_params' in st.session_state:
                display_latex_formula_detailed(
                    study_design,
                    outcome_type,
                    st.session_state.posthoc_params,
                    is_posthoc=True
                )
            else:
                st.info("Please calculate post-hoc power first to view the formula.")
        
        with posthoc_tab3:
            if 'posthoc_results' in st.session_state:
                # Create post-hoc specific visualizations 
                create_posthoc_visualizations(
                    study_design,
                    outcome_type, 
                    st.session_state.posthoc_params,
                    st.session_state.posthoc_results
                )
                
            else:
                st.info("Please calculate post-hoc power first to view the interpretation and visualizations.")
    
    with main_tab3:
        st.markdown('<div class="education-container">', unsafe_allow_html=True)
        st.markdown("### 📚 **Educational Resources & Statistical Guide**")
        
        edu_tab1, edu_tab2, edu_tab3, edu_tab4 = st.tabs(["📖 Basics", "🧮 Formulas", "📊 Examples", "📚 References"])
        
        with edu_tab1:
            st.markdown("""
            ## **Statistical Power Analysis Fundamentals**
            
            ### **Key Concepts**
            
            **Statistical Power (1-β)**
            - The probability of correctly rejecting a false null hypothesis
            - Commonly set to 80% (0.80) in medical research
            - Higher power requires larger sample sizes
            
            **Type I Error (α)**
            - The probability of incorrectly rejecting a true null hypothesis (false positive)
            - Commonly set to 5% (0.05) in medical research
            - Lower α requires larger sample sizes
            
            **Effect Size**
            - The magnitude of difference between groups
            - Cohen's d for continuous outcomes: Small (0.2), Medium (0.5), Large (0.8)
            - Proportion difference for dichotomous outcomes
            
            **Sample Size Determinants**
            1. Desired power (1-β)
            2. Significance level (α)  
            3. Effect size
            4. Variability (standard deviation)
            5. Allocation ratio between groups
            """)
        
        with edu_tab2:
            st.markdown("""
            ## **Statistical Formulas Reference**
            
            ### **Sample Size Calculations**
            
            **Two-Sample T-Test (Continuous)**
            """)
            st.latex(r'''n_1 = \frac{(z_{1-\alpha/2} + z_{1-\beta})^2 (\sigma_1^2 + \sigma_2^2/k)}{(\mu_1 - \mu_2)^2}''')
            
            st.markdown("**Two-Proportion Test (Dichotomous)**")
            st.latex(r'''n = \frac{[z_{1-\alpha/2}\sqrt{\bar{p}\bar{q}(1 + \frac{1}{k})} + z_{1-\beta}\sqrt{p_1q_1 + \frac{p_2q_2}{k}}]^2}{(p_1 - p_2)^2}''')
            
            st.markdown("**One-Sample Proportion Test**")
            st.latex(r'''N = \frac{[z_{1-\alpha/2}\sqrt{p_0q_0} + z_{1-\beta}\sqrt{p_1q_1}]^2}{(p_1 - p_0)^2}''')
        
        with edu_tab3:
            st.markdown("""
            ## **Clinical Research Examples**
            
            ### **Example 1: Drug Efficacy Trial**
            - **Study**: Compare new drug vs. placebo for hypertension
            - **Outcome**: Proportion achieving target blood pressure
            - **Parameters**: Control 40%, Treatment 60%, α=0.05, Power=80%
            - **Result**: ~97 subjects per group (194 total)
            
            ### **Example 2: Diagnostic Test Accuracy**
            - **Study**: New test vs. known population sensitivity
            - **Outcome**: Sensitivity of diagnostic test
            - **Parameters**: Population 70%, Expected 85%, α=0.05, Power=80%
            - **Result**: ~118 subjects needed
            
            ### **Example 3: Quality Improvement**
            - **Study**: Before/after intervention comparison
            - **Outcome**: Mean satisfaction score
            - **Parameters**: Before=7.2, After=8.0, SD=1.5, α=0.05, Power=80%
            - **Result**: ~45 subjects per group (90 total)
            """)
        
        with edu_tab4:
            st.markdown("""
            ## **References & Further Reading**
            
            ### **Primary Statistical References**
            1. **Chow, S.C., Shao, J., Wang, H., Lokhnygina, Y.** (2017). *Sample Size Calculations in Clinical Research* (3rd ed.). Chapman & Hall/CRC.
            
            2. **Cohen, J.** (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Lawrence Erlbaum Associates.
            
            3. **Fleiss, J.L., Levin, B., Paik, M.C.** (2003). *Statistical Methods for Rates and Proportions* (3rd ed.). John Wiley & Sons.
            
            ### **Post-Hoc Power Analysis Critiques**
            4. **Levine, M., Ensom, M.H.** (2001). Post hoc power analysis: an idea whose time has passed? *Pharmacotherapy*, 21(4), 405-409.
            
            5. **Hoenig, J.M., Heisey, D.M.** (2001). The abuse of power: the pervasive fallacy of power calculations for data analysis. *The American Statistician*, 55(1), 19-24.
            
            ### **Online Resources**
            - **FDA Guidance for Industry**: Statistical Approaches to Establishing Bioequivalence
            - **ICH E9 Guideline**: Statistical Principles for Clinical Trials
            - **CONSORT Statement**: Standards for reporting clinical trials
            """)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6c757d; font-size: 0.9rem;">
        🏥 Professional Clinical Research Tool • Created for Healthcare Professionals • Always Consult Statistician for Complex Studies
        <br><br>
        📚 <strong>Educational Resources:</strong> This calculator uses established statistical formulas from Chow, Shao, Wang & Lokhnygina's 
        "Sample Size Calculations in Clinical Research" and Cohen's "Statistical Power Analysis for the Behavioral Sciences"
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
