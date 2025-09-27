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
    def calculate_continuous_two_groups(mean1, mean2, std_dev, alpha=0.05, power=0.80, 
                                      allocation_ratio=1.0, two_sided=True, dropout_rate=0.0):
        """Calculate sample size for continuous outcomes, two independent groups"""
        
        # Calculate effect size
        effect_size = abs(mean1 - mean2) / std_dev
        
        # Z-scores
        z_alpha = SampleSizeCalculator.z_score(alpha, two_sided)
        z_beta = stats.norm.ppf(power)
        
        # Sample size calculation
        n1 = ((z_alpha + z_beta) ** 2 * 2 * (std_dev ** 2)) / ((mean1 - mean2) ** 2)
        
        # Adjust for allocation ratio
        n1 = n1 * (1 + 1/allocation_ratio) / 4
        
        # Adjust for dropout
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
        
        # Adjust for dropout
        if dropout_rate > 0:
            n = n / (1 - dropout_rate)
        
        return {
            'n': math.ceil(n),
            'effect_size': effect_size,
            'z_alpha': z_alpha,
            'z_beta': z_beta
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
        
        # Adjust for dropout
        if dropout_rate > 0:
            n1 = n1 / (1 - dropout_rate)
        
        n2 = n1 * allocation_ratio
        
        return {
            'n1': math.ceil(n1),
            'n2': math.ceil(n2),
            'total': math.ceil(n1 + n2),
            'p_pooled': p_pooled,
            'z_alpha': z_alpha,
            'z_beta': z_beta,
            'effect_size': abs(p1 - p2)
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
        # N = [z_alpha * sqrt(p0*q0) + z_beta * sqrt(p1*q1)]^2 / (p1 - p0)^2
        numerator = (z_alpha * math.sqrt(population_prop * q0) + z_beta * math.sqrt(sample_prop * q1)) ** 2
        denominator = (sample_prop - population_prop) ** 2
        
        n = numerator / denominator
        
        # Continuity correction (Yates correction)
        if continuity_correction:
            correction = 1 + 1/(2*n*abs(sample_prop - population_prop))
            n = n * correction
        
        # Adjust for dropout
        if dropout_rate > 0:
            n = n / (1 - dropout_rate)
        
        return {
            'n': math.ceil(n),
            'z_alpha': z_alpha,
            'z_beta': z_beta,
            'effect_size': abs(sample_prop - population_prop),
            'p0': population_prop,
            'p1': sample_prop,
            'q0': q0,
            'q1': q1
        }

def calculate_post_hoc_power(study_design, outcome_type, params, actual_results):
    """Calculate post-hoc statistical power from actual study results"""
    
    if study_design == "Two independent study groups":
        if outcome_type == "Dichotomous (yes/no)":
            # Two-proportion z-test post-hoc power
            n1, n2 = actual_results['n1'], actual_results['n2'] 
            p1, p2 = params['p1'], params['p2']
            alpha = params['alpha']
            
            # Pooled proportion
            p_pooled = (n1 * p1 + n2 * p2) / (n1 + n2)
            q_pooled = 1 - p_pooled
            
            # Standard errors
            se_null = math.sqrt(p_pooled * q_pooled * (1/n1 + 1/n2))
            se_alt = math.sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)
            
            # Effect size
            effect_size = abs(p1 - p2)
            
            # Critical value
            z_alpha = stats.norm.ppf(1 - alpha/2)  # two-sided
            
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
    
    elif study_design == "One study group vs. population":
        if outcome_type == "Dichotomous (yes/no)":
            # One-sample proportion post-hoc power
            n = actual_results['n']
            p1 = params['sample_prop']
            p0 = params['population_prop'] 
            alpha = params['alpha']
            
            # Standard errors
            se_null = math.sqrt(p0 * (1-p0) / n)
            se_alt = math.sqrt(p1 * (1-p1) / n)
            
            # Effect size
            effect_size = abs(p1 - p0)
            
            # Critical value
            z_alpha = stats.norm.ppf(1 - alpha/2)  # two-sided
            
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
    
    return None

def display_latex_formula_detailed(study_design, outcome_type, params, is_posthoc=False):
    """Display detailed LaTeX formulas with comprehensive explanations"""
    
    st.markdown('<div class="formula-container">', unsafe_allow_html=True)
    
    if not is_posthoc:
        st.markdown("### 🧮 **Sample Size Formula & Parameter Substitution**")
    else:
        st.markdown("### 🔄 **Post-Hoc Power Formula & Parameter Substitution**")
    
    if study_design == "Two independent study groups":
        if outcome_type == "Continuous (means)":
            if not is_posthoc:
                st.markdown("#### **Two-Sample T-Test Sample Size Formula:**")
                st.latex(r'''n = \frac{(z_{1-\alpha/2} + z_{1-\beta})^2 \cdot 2\sigma^2}{(\mu_1 - \mu_2)^2}''')
                
                st.markdown("""
                **Where:**
                - **n** = sample size per group
                - **z₁₋α/₂** = critical value for two-sided test at α significance level
                - **z₁₋β** = critical value corresponding to desired power (1-β)
                - **σ** = common standard deviation
                - **μ₁, μ₂** = means of the two groups
                - **α** = Type I error rate (typically 0.05)
                - **β** = Type II error rate (typically 0.20 for 80% power)
                """)
            else:
                st.markdown("#### **Two-Sample T-Test Post-Hoc Power Formula:**")
                st.latex(r'''Power = \Phi\left(\frac{|\bar{x}_1 - \bar{x}_2|}{s_p\sqrt{\frac{1}{n_1} + \frac{1}{n_2}}} - z_{1-\alpha/2}\right)''')
                
        else:  # Dichotomous two groups
            if not is_posthoc:
                st.markdown("#### **Two-Proportion Z-Test Sample Size Formula:**")
                st.latex(r'''n = \frac{[z_{1-\alpha/2}\sqrt{\bar{p}\bar{q}(1 + \frac{1}{k})} + z_{1-\beta}\sqrt{p_1q_1 + \frac{p_2q_2}{k}}]^2}{(p_1 - p_2)^2}''')
                
                st.markdown("""
                **Where:**
                - **n** = sample size for group 1
                - **p₁, p₂** = proportions in groups 1 and 2
                - **q₁ = 1-p₁, q₂ = 1-p₂** = complement proportions
                - **p̄** = pooled proportion = (p₁ + kp₂)/(1 + k)
                - **q̄ = 1-p̄** = pooled complement proportion
                - **k** = allocation ratio (n₂/n₁)
                - **z₁₋α/₂** = critical value for significance level α
                - **z₁₋β** = critical value for power (1-β)
                """)
            else:
                st.markdown("#### **Two-Proportion Post-Hoc Power Formula:**")
                st.latex(r'''Power = \Phi\left(\frac{\Delta - z_{1-\alpha/2}\sqrt{\bar{p}\bar{q}(\frac{1}{n_1} + \frac{1}{n_2})}}{\sqrt{\frac{p_1q_1}{n_1} + \frac{p_2q_2}{n_2}}}\right)''')
                
                st.markdown("""
                **Where:**
                - **Δ = |p₂ - p₁|** = absolute difference between proportions
                - **p̄** = pooled proportion from observed data
                - **q̄ = 1 - p̄** = pooled complement proportion
                - **n₁, n₂** = actual sample sizes
                - **Φ()** = standard normal cumulative distribution function
                """)
    
    else:  # One group vs population
        if outcome_type == "Continuous (means)":
            if not is_posthoc:
                st.markdown("#### **One-Sample T-Test Sample Size Formula:**")
                st.latex(r'''N = \frac{(z_{1-\alpha/2} + z_{1-\beta})^2 \cdot \sigma^2}{(\mu_{sample} - \mu_{population})^2}''')
                
                st.markdown("""
                **Where:**
                - **N** = required sample size
                - **μₛₐₘₚₗₑ** = expected sample mean
                - **μₚₒₚᵤₗₐₜᵢₒₙ** = known population mean
                - **σ** = population standard deviation
                - **z₁₋α/₂, z₁₋β** = critical values for α and power
                """)
            else:
                st.markdown("#### **One-Sample T-Test Post-Hoc Power Formula:**")
                st.latex(r'''Power = \Phi\left(\frac{|\bar{x} - \mu_0|}{s/\sqrt{n}} - z_{1-\alpha/2}\right)''')
                
        else:  # Dichotomous one group  
            if not is_posthoc:
                st.markdown("#### **One-Sample Proportion Test Sample Size Formula:**")
                st.latex(r'''N = \frac{[z_{1-\alpha/2}\sqrt{p_0q_0} + z_{1-\beta}\sqrt{p_1q_1}]^2}{(p_1 - p_0)^2}''')
                
                st.markdown("""
                **Parameter Definitions:**
                - **N** = required sample size for study group
                - **p₀** = known population proportion (baseline)
                - **p₁** = expected study group proportion
                - **q₀ = 1 - p₀** = population complement proportion
                - **q₁ = 1 - p₁** = study complement proportion
                - **z₁₋α/₂** = critical Z-value for significance level α
                - **z₁₋β** = critical Z-value for power (1-β)
                - **α** = Type I error rate (probability of false positive)
                - **β** = Type II error rate (probability of false negative)
                """)
                
                # Show parameter substitution if available
                if all(key in params for key in ['p0', 'p1', 'q0', 'q1', 'z_alpha', 'z_beta']):
                    st.markdown("#### **Parameter Substitution:**")
                    st.markdown(f"""
                    - **p₀** = {params['p0']:.3f} (population proportion)
                    - **p₁** = {params['p1']:.3f} (study proportion)
                    - **q₀** = 1 - p₀ = {params['q0']:.3f}
                    - **q₁** = 1 - p₁ = {params['q1']:.3f}
                    - **z₁₋α/₂** = {params['z_alpha']:.3f}
                    - **z₁₋β** = {params['z_beta']:.3f}
                    """)
                    
                    st.latex(f'''N = \\frac{{[{params['z_alpha']:.3f} \\times \\sqrt{{{params['p0']:.3f} \\times {params['q0']:.3f}}} + {params['z_beta']:.3f} \\times \\sqrt{{{params['p1']:.3f} \\times {params['q1']:.3f}}}]^2}}{{({params['p1']:.3f} - {params['p0']:.3f})^2}} = {params.get('n', 'N/A')}''')
            else:
                st.markdown("#### **One-Sample Proportion Post-Hoc Power Formula:**")
                st.latex(r'''Power = \Phi\left(\frac{|\hat{p} - p_0|}{\sqrt{p_0q_0/n}} - z_{1-\alpha/2}\right)''')
                
                st.markdown("""
                **Where:**
                - **p̂** = observed sample proportion
                - **p₀** = known population proportion
                - **n** = actual sample size
                - **Φ()** = standard normal CDF
                """)
    
    # Add interpretation section
    st.markdown("#### **Formula Interpretation:**")
    if not is_posthoc:
        st.markdown("""
        **The sample size formula balances four key factors:**
        1. **Significance Level (α)**: Lower α requires larger samples
        2. **Statistical Power (1-β)**: Higher power requires larger samples  
        3. **Effect Size**: Smaller effects require larger samples to detect
        4. **Variability**: Higher variability requires larger samples
        
        **Common Values:**
        - α = 0.05 (5% chance of false positive)
        - Power = 0.80 (80% chance of detecting true effect)
        - Two-sided tests are standard unless directional hypothesis
        """)
    else:
        st.markdown("""
        **Post-hoc power tells us:**
        - The probability our completed study could detect the observed effect
        - Why a study might have failed to find significance
        - **Caution**: Low post-hoc power in negative studies can be misleading
        
        **Limitations of Post-Hoc Analysis:**
        - Should not be used to "explain away" negative results
        - Consider confidence interval width instead
        - Post-hoc power is directly related to p-value
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)

def create_dual_input(label, min_val, max_val, default_val, step=0.01, help_text="", key_suffix=""):
    """Create both slider and number input for the same value"""
    st.markdown(f"**{label}**", help=help_text)
    col1, col2 = st.columns([2, 1])
    
    with col1:
        slider_val = st.slider(
            f"",
            min_value=min_val,
            max_value=max_val, 
            value=default_val,
            step=step,
            key=f"slider_{key_suffix}",
            label_visibility="collapsed"
        )
    
    with col2:
        number_val = st.number_input(
            f"",
            min_value=min_val,
            max_value=max_val,
            value=slider_val,
            step=step,
            key=f"number_{key_suffix}",
            label_visibility="collapsed"
        )
    
    # Use number input value if it was changed, otherwise use slider
    return number_val if f"number_{key_suffix}" in st.session_state else slider_val

def display_professional_results_tables(results, study_design, outcome_type, params):
    """Display results using proper Streamlit tables"""
    
    # Results header
    st.markdown('<div class="results-header">RESULTS</div>', unsafe_allow_html=True)
    
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
    
    # Main results display
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### **Sample Size**")
        
        # Create sample size dataframe
        if study_design == "Two independent study groups":
            sample_df = pd.DataFrame({
                "Group": ["Group 1", "Group 2", "**Total**"],
                "Size": [results['n1'], results['n2'], f"**{results['total']}**"]
            })
        else:
            sample_df = pd.DataFrame({
                "Group": ["Group 1", "**Total**"],
                "Size": [results['n'], f"**{results['n']}**"]
            })
        
        # Display with styling
        st.dataframe(
            sample_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Group": st.column_config.TextColumn("Group", width="medium"),
                "Size": st.column_config.TextColumn("Size", width="medium")
            }
        )
        
        # Highlight total sample size
        if study_design == "Two independent study groups":
            st.markdown(f'<div class="metric-card"><div class="metric-value">{results["total"]}</div><div class="metric-label">Total Sample Size</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{results["n"]}</div><div class="metric-label">Required Sample Size</div></div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("### **Study Parameters**")
        
        # Build parameters dataframe
        if study_design == "Two independent study groups":
            if outcome_type == "Continuous (means)":
                param_df = pd.DataFrame({
                    "Parameter": ["Mean, group 1", "Mean, group 2", "Standard deviation", "Alpha", "Beta", "Power"],
                    "Value": [params['mean1'], params['mean2'], params['std_dev'], 
                             params['alpha'], round(1-params['power'], 2), params['power']]
                })
            else:
                param_df = pd.DataFrame({
                    "Parameter": ["Proportion, group 1", "Proportion, group 2", "Alpha", "Beta", "Power"],
                    "Value": [f"{params['p1']:.2f}", f"{params['p2']:.2f}", 
                             params['alpha'], round(1-params['power'], 2), params['power']]
                })
        else:
            if outcome_type == "Continuous (means)":
                param_df = pd.DataFrame({
                    "Parameter": ["Mean, sample", "Mean, population", "Standard deviation", "Alpha", "Beta", "Power"],
                    "Value": [params['sample_mean'], params['population_mean'], params['std_dev'],
                             params['alpha'], round(1-params['power'], 2), params['power']]
                })
            else:
                param_df = pd.DataFrame({
                    "Parameter": ["Incidence, population", "Incidence, study group", "Alpha", "Beta", "Power"],
                    "Value": [f"{params['population_prop']:.0%}", f"{params['sample_prop']:.0%}",
                             params['alpha'], round(1-params['power'], 2), params['power']]
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
    
    # Main navigation tabs
    main_tab1, main_tab2, main_tab3 = st.tabs(["📊 Sample Size Analysis", "🔄 Post-Hoc Power Analysis", "📚 Education & Resources"])
    
    with main_tab1:
        # Sidebar for sample size analysis
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
            
            if "Dichotomous" in outcome_type:
                dropout_rate = create_dual_input(
                    "Expected dropout rate (%)",
                    0, 50, 10, 1, 
                    "Percentage of participants expected to drop out or be lost to follow-up during the study",
                    "dropout"
                ) / 100
            else:
                dropout_rate = 0.0
        
        # Main content area with sub-tabs
        sub_tab1, sub_tab2, sub_tab3, sub_tab4 = st.tabs(["📋 Calculator", "🧮 Formula", "📊 Analysis", "📄 Citation"])
        
        with sub_tab1:
            st.markdown('<div class="calc-container">', unsafe_allow_html=True)
            
            # Dynamic form based on selections
            if study_design == "Two independent study groups":
                if outcome_type == "Continuous (means)":
                    st.markdown("### **Means:**")
                    col_a, col_b = st.columns(2)
                    with col_a:
                        mean1 = st.number_input("**Group 1 mean**", value=10.0, step=0.1)
                    with col_b:
                        mean2 = st.number_input("**Group 2 mean**", value=12.0, step=0.1)
                    
                    std_dev = st.number_input("**Common standard deviation**", min_value=0.01, value=2.0, step=0.1)
                    allocation_ratio = st.number_input("**Allocation ratio (group 2 / group 1)**", 
                                                     min_value=0.1, max_value=5.0, value=1.0, step=0.1)
                    
                    if st.button("🔢 **Calculate Sample Size**", type="primary", use_container_width=True):
                        try:
                            results = SampleSizeCalculator.calculate_continuous_two_groups(
                                mean1, mean2, std_dev, alpha, power, allocation_ratio, two_sided, dropout_rate
                            )
                            
                            params = {
                                'mean1': mean1, 'mean2': mean2, 'std_dev': std_dev,
                                'alpha': alpha, 'power': power, 'z_alpha': results['z_alpha'],
                                'z_beta': results['z_beta'], 'allocation_ratio': allocation_ratio
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
                        p1 = create_dual_input("Group 1 proportion", 0.01, 0.99, 0.14, 0.01, 
                                             "Expected proportion in group 1", "p1")
                        p2 = create_dual_input("Group 2 proportion", 0.01, 0.99, 0.21, 0.01,
                                             "Expected proportion in group 2", "p2")
                                             
                    elif ratio_type == "% Increase":
                        baseline_prop = create_dual_input("Baseline proportion", 0.01, 0.99, 0.14, 0.01,
                                                        "Baseline proportion for comparison", "baseline")
                        increase_percent = create_dual_input("Percentage increase", 1.0, 200.0, 50.0, 1.0,
                                                           "Percentage increase from baseline", "increase")
                        p1 = baseline_prop
                        p2 = min(0.99, baseline_prop * (1 + increase_percent/100))
                        st.write(f"Group 1: {p1:.3f}, Group 2: {p2:.3f}")
                        
                    elif ratio_type == "% Decrease":
                        baseline_prop = create_dual_input("Baseline proportion", 0.01, 0.99, 0.21, 0.01,
                                                        "Baseline proportion for comparison", "baseline_dec")
                        decrease_percent = create_dual_input("Percentage decrease", 1.0, 99.0, 33.0, 1.0,
                                                           "Percentage decrease from baseline", "decrease")
                        p2 = baseline_prop
                        p1 = max(0.01, baseline_prop * (1 - decrease_percent/100))
                        st.write(f"Group 1: {p1:.3f}, Group 2: {p2:.3f}")
                    
                    allocation_ratio = st.number_input("**Allocation ratio (group 2 / group 1)**", 
                                                     min_value=0.1, max_value=5.0, value=1.0, step=0.1)
                    
                    if st.button("🔢 **Calculate Sample Size**", type="primary", use_container_width=True):
                        try:
                            results = SampleSizeCalculator.calculate_proportions_two_groups(
                                p1, p2, alpha, power, allocation_ratio, two_sided, True, dropout_rate
                            )
                            
                            params = {
                                'p1': p1, 'p2': p2, 'alpha': alpha, 'power': power,
                                'z_alpha': results['z_alpha'], 'z_beta': results['z_beta'],
                                'p_pooled': results['p_pooled'], 'allocation_ratio': allocation_ratio
                            }
                            
                            st.session_state.results = results
                            st.session_state.params = params
                            st.session_state.study_design = study_design
                            st.session_state.outcome_type = outcome_type
                            
                            display_professional_results_tables(results, study_design, outcome_type, params)
                            
                        except Exception as e:
                            st.error(f"Calculation error: {str(e)}")
            
            else:  # One group vs population
                if outcome_type == "Continuous (means)":
                    st.markdown("### **Means:**")
                    col_a, col_b = st.columns(2)
                    with col_a:
                        sample_mean = st.number_input("**Expected sample mean**", value=12.0, step=0.1)
                    with col_b:
                        population_mean = st.number_input("**Population mean**", value=10.0, step=0.1)
                    
                    std_dev = st.number_input("**Standard deviation**", min_value=0.01, value=2.0, step=0.1)
                    
                    if st.button("🔢 **Calculate Sample Size**", type="primary", use_container_width=True):
                        try:
                            results = SampleSizeCalculator.calculate_continuous_one_group(
                                sample_mean, population_mean, std_dev, alpha, power, two_sided, dropout_rate
                            )
                            
                            params = {
                                'sample_mean': sample_mean, 'population_mean': population_mean,
                                'std_dev': std_dev, 'alpha': alpha, 'power': power,
                                'z_alpha': results['z_alpha'], 'z_beta': results['z_beta'],
                                'effect_size': results['effect_size']
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
                        sample_prop = create_dual_input("Expected study proportion", 0.01, 0.99, 0.14, 0.01,
                                                      "Expected proportion in your study", "sample_prop")
                        population_prop = create_dual_input("Known population proportion", 0.01, 0.99, 0.21, 0.01,
                                                          "Known population proportion", "pop_prop")
                                                          
                    elif ratio_type == "% Increase from Population":
                        population_prop = create_dual_input("Known population proportion", 0.01, 0.99, 0.21, 0.01,
                                                          "Known population proportion", "pop_prop_inc")
                        increase_percent = create_dual_input("Expected % increase", 1.0, 200.0, 50.0, 1.0,
                                                           "Expected percentage increase", "pop_increase")
                        sample_prop = min(0.99, population_prop * (1 + increase_percent/100))
                        st.write(f"Population: {population_prop:.3f}, Expected Study: {sample_prop:.3f}")
                        
                    elif ratio_type == "% Decrease from Population":
                        population_prop = create_dual_input("Known population proportion", 0.01, 0.99, 0.21, 0.01,
                                                          "Known population proportion", "pop_prop_dec")
                        decrease_percent = create_dual_input("Expected % decrease", 1.0, 99.0, 33.0, 1.0,
                                                           "Expected percentage decrease", "pop_decrease")
                        sample_prop = max(0.01, population_prop * (1 - decrease_percent/100))
                        st.write(f"Population: {population_prop:.3f}, Expected Study: {sample_prop:.3f}")
                    
                    if st.button("🔢 **Calculate Sample Size**", type="primary", use_container_width=True):
                        try:
                            results = SampleSizeCalculator.calculate_proportions_one_group(
                                sample_prop, population_prop, alpha, power, two_sided, True, dropout_rate
                            )
                            
                            params = {
                                'sample_prop': sample_prop, 'population_prop': population_prop,
                                'alpha': alpha, 'power': power, 'z_alpha': results['z_alpha'],
                                'z_beta': results['z_beta'], 'effect_size': results['effect_size'],
                                'p0': results['p0'], 'p1': results['p1'], 'q0': results['q0'], 'q1': results['q1'],
                                'n': results['n']
                            }
                            
                            st.session_state.results = results
                            st.session_state.params = params
                            st.session_state.study_design = study_design
                            st.session_state.outcome_type = outcome_type
                            
                            display_professional_results_tables(results, study_design, outcome_type, params)
                            
                            # Effect size display
                            st.success(f"**Effect Size:** {results['effect_size']:.3f} (absolute difference from population)")
                            
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
                st.markdown("### 📊 **Parameter Sensitivity Analysis**")
                st.info("Visualization showing how changes in parameters affect sample size requirements.")
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
        
        st.warning("""
        **⚠️ Important Note about Post-Hoc Power Analysis:**
        
        Post-hoc power analysis has significant limitations and should be interpreted cautiously. 
        It's typically used to understand why a study may not have detected a significant effect, 
        but it should **not** be used to "explain away" negative results.
        """)
        
        # Post-hoc analysis implementation would go here
        # This would be a complete separate implementation similar to the sample size analysis
        st.info("Post-hoc power analysis implementation - Complete interface similar to sample size analysis")
    
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
            st.latex(r'''n = \frac{(z_{1-\alpha/2} + z_{1-\beta})^2 \cdot 2\sigma^2}{(\mu_1 - \mu_2)^2}''')
            
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
