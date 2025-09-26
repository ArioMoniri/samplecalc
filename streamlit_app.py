import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats as stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Clinical Sample Size Calculator",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for medical/clinical styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 2rem;
        border-bottom: 3px solid #3b82f6;
        padding-bottom: 1rem;
    }
    
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1e40af;
        margin: 1.5rem 0 1rem 0;
        border-left: 4px solid #3b82f6;
        padding-left: 1rem;
    }
    
    .formula-box {
        background: #f8fafc;
        border: 2px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .result-box {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border: 2px solid #0284c7;
        border-radius: 12px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .warning-box {
        background: #fef3c7;
        border: 2px solid #f59e0b;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .info-box {
        background: #ecfdf5;
        border: 2px solid #10b981;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .citation-box {
        background: #f1f5f9;
        border: 1px solid #cbd5e1;
        border-radius: 6px;
        padding: 1rem;
        font-family: monospace;
        font-size: 0.9rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Statistical calculation functions
@st.cache_data
def calculate_sample_size_two_proportions(p1, p2, alpha=0.05, beta=0.2, ratio=1, two_sided=True):
    """Calculate sample size for comparing two proportions."""
    if two_sided:
        z_alpha = stats.norm.ppf(1 - alpha/2)
    else:
        z_alpha = stats.norm.ppf(1 - alpha)
    
    z_beta = stats.norm.ppf(1 - beta)
    
    # Pooled proportion
    p_bar = (p1 + ratio * p2) / (1 + ratio)
    q_bar = 1 - p_bar
    
    # Effect size
    effect_size = abs(p2 - p1)
    
    if effect_size == 0:
        return float('inf'), float('inf')
    
    # Sample size calculation
    numerator = (z_alpha * np.sqrt(p_bar * q_bar * (1 + 1/ratio)) + 
                z_beta * np.sqrt(p1 * (1-p1) + p2 * (1-p2) / ratio))**2
    
    n1 = numerator / (effect_size**2)
    n2 = n1 * ratio
    
    return math.ceil(n1), math.ceil(n2)

@st.cache_data
def calculate_sample_size_two_means(mu1, mu2, sigma, alpha=0.05, beta=0.2, ratio=1, two_sided=True):
    """Calculate sample size for comparing two means."""
    if two_sided:
        z_alpha = stats.norm.ppf(1 - alpha/2)
    else:
        z_alpha = stats.norm.ppf(1 - alpha)
    
    z_beta = stats.norm.ppf(1 - beta)
    
    effect_size = abs(mu2 - mu1)
    
    if effect_size == 0:
        return float('inf'), float('inf')
    
    # Sample size calculation
    n1 = 2 * (sigma**2) * ((z_alpha + z_beta)**2) / (effect_size**2)
    n2 = n1 * ratio
    
    return math.ceil(n1), math.ceil(n2)

@st.cache_data
def calculate_sample_size_one_proportion(p, p0, alpha=0.05, beta=0.2, two_sided=True):
    """Calculate sample size for one proportion vs population."""
    if two_sided:
        z_alpha = stats.norm.ppf(1 - alpha/2)
    else:
        z_alpha = stats.norm.ppf(1 - alpha)
    
    z_beta = stats.norm.ppf(1 - beta)
    
    effect_size = abs(p - p0)
    
    if effect_size == 0:
        return float('inf')
    
    # Sample size calculation
    n = ((z_alpha * np.sqrt(p0 * (1-p0)) + z_beta * np.sqrt(p * (1-p)))**2) / (effect_size**2)
    
    return math.ceil(n)

@st.cache_data
def calculate_sample_size_one_mean(mu, mu0, sigma, alpha=0.05, beta=0.2, two_sided=True):
    """Calculate sample size for one mean vs population."""
    if two_sided:
        z_alpha = stats.norm.ppf(1 - alpha/2)
    else:
        z_alpha = stats.norm.ppf(1 - alpha)
    
    z_beta = stats.norm.ppf(1 - beta)
    
    effect_size = abs(mu - mu0)
    
    if effect_size == 0:
        return float('inf')
    
    # Sample size calculation
    n = ((z_alpha + z_beta)**2 * sigma**2) / (effect_size**2)
    
    return math.ceil(n)

def generate_power_curve(effect_sizes, n, alpha, study_type, outcome_type, **params):
    """Generate power curve for visualization."""
    powers = []
    
    for effect in effect_sizes:
        if study_type == "Two independent study groups":
            if outcome_type == "Dichotomous (yes/no)":
                p1, p2_base, ratio = params['p1'], params['p2'], params.get('ratio', 1)
                p2 = p1 + effect
                if 0 <= p2 <= 1:
                    # Calculate power for given n
                    z_alpha = stats.norm.ppf(1 - alpha/2)
                    p_bar = (p1 + ratio * p2) / (1 + ratio)
                    se_null = np.sqrt(p_bar * (1-p_bar) * (1 + 1/ratio) / n)
                    se_alt = np.sqrt(p1 * (1-p1) / n + p2 * (1-p2) / (n * ratio))
                    z_stat = effect / se_alt
                    power = 1 - stats.norm.cdf(z_alpha - z_stat) + stats.norm.cdf(-z_alpha - z_stat)
                else:
                    power = 0
            else:  # Continuous
                sigma = params['sigma']
                z_alpha = stats.norm.ppf(1 - alpha/2)
                z_beta = (effect * np.sqrt(n/2)) / sigma - z_alpha
                power = 1 - stats.norm.cdf(z_beta)
        else:  # One group vs population
            if outcome_type == "Dichotomous (yes/no)":
                p, p0 = params['p'], params['p0']
                p_alt = p0 + effect
                if 0 <= p_alt <= 1:
                    z_alpha = stats.norm.ppf(1 - alpha/2)
                    se_null = np.sqrt(p0 * (1-p0) / n)
                    se_alt = np.sqrt(p_alt * (1-p_alt) / n)
                    z_stat = effect / se_alt
                    power = 1 - stats.norm.cdf(z_alpha - z_stat) + stats.norm.cdf(-z_alpha - z_stat)
                else:
                    power = 0
            else:  # Continuous
                sigma = params['sigma']
                z_alpha = stats.norm.ppf(1 - alpha/2)
                z_beta = (effect * np.sqrt(n)) / sigma - z_alpha
                power = 1 - stats.norm.cdf(z_beta)
        
        powers.append(max(0, min(1, power)))
    
    return powers

# Main application
def main():
    st.markdown('<h1 class="main-header">🔬 Clinical Sample Size Calculator</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #64748b;">Professional statistical tool for clinical research sample size determination</p>', unsafe_allow_html=True)
    
    # Sidebar for main controls
    st.sidebar.markdown("## 📊 Study Configuration")
    
    # Study design selection
    study_type = st.sidebar.radio(
        "Select Study Design:",
        ["Two independent study groups", "One study group vs. population"],
        help="Choose between comparing two groups or comparing one group to a known population parameter"
    )
    
    # Outcome type selection
    outcome_type = st.sidebar.radio(
        "Select Outcome Type:",
        ["Dichotomous (yes/no)", "Continuous (means)"],
        help="Dichotomous: binary outcomes (success/failure). Continuous: measured values (height, weight, etc.)"
    )
    
    # Statistical parameters
    st.sidebar.markdown("### 🎯 Statistical Parameters")
    
    confidence_level = st.sidebar.selectbox(
        "Confidence Level (%)",
        [90, 95, 99],
        index=1,
        help="Probability of avoiding Type I error (false positive)"
    )
    alpha = (100 - confidence_level) / 100
    
    power = st.sidebar.selectbox(
        "Statistical Power (%)",
        [80, 85, 90, 95],
        index=0,
        help="Probability of detecting a true effect (1 - Type II error)"
    )
    beta = (100 - power) / 100
    
    two_sided = st.sidebar.checkbox(
        "Two-sided test",
        value=True,
        help="Two-sided: effect can be in either direction. One-sided: effect in specific direction only"
    )
    
    # Create main content area with tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Calculator", "📊 Formula Details", "📈 Power Analysis", "📚 Educational Content"])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<div class="section-header">Study Parameters</div>', unsafe_allow_html=True)
            
            # Dynamic form based on selections
            if study_type == "Two independent study groups":
                if outcome_type == "Dichotomous (yes/no)":
                    st.markdown("**Group Proportions:**")
                    p1 = st.number_input(
                        "Group 1 expected proportion",
                        min_value=0.0, max_value=1.0, value=0.5, step=0.01,
                        help="Expected proportion of events in the first group"
                    )
                    p2 = st.number_input(
                        "Group 2 expected proportion", 
                        min_value=0.0, max_value=1.0, value=0.3, step=0.01,
                        help="Expected proportion of events in the second group"
                    )
                    
                    ratio = st.number_input(
                        "Allocation ratio (Group 2 : Group 1)",
                        min_value=0.1, max_value=10.0, value=1.0, step=0.1,
                        help="Ratio of sample sizes. 1.0 means equal groups"
                    )
                    
                    # Calculate sample size
                    try:
                        n1, n2 = calculate_sample_size_two_proportions(p1, p2, alpha, beta, ratio, two_sided)
                        params = {'p1': p1, 'p2': p2, 'ratio': ratio}
                    except:
                        n1, n2 = 0, 0
                        params = {}
                
                else:  # Continuous
                    st.markdown("**Group Means and Standard Deviation:**")
                    mu1 = st.number_input(
                        "Group 1 expected mean",
                        value=100.0, step=0.1,
                        help="Expected mean value in the first group"
                    )
                    mu2 = st.number_input(
                        "Group 2 expected mean",
                        value=110.0, step=0.1,
                        help="Expected mean value in the second group"
                    )
                    sigma = st.number_input(
                        "Standard deviation",
                        min_value=0.01, value=15.0, step=0.1,
                        help="Common standard deviation (assumed equal in both groups)"
                    )
                    
                    ratio = st.number_input(
                        "Allocation ratio (Group 2 : Group 1)",
                        min_value=0.1, max_value=10.0, value=1.0, step=0.1,
                        help="Ratio of sample sizes. 1.0 means equal groups"
                    )
                    
                    # Calculate sample size
                    try:
                        n1, n2 = calculate_sample_size_two_means(mu1, mu2, sigma, alpha, beta, ratio, two_sided)
                        params = {'mu1': mu1, 'mu2': mu2, 'sigma': sigma, 'ratio': ratio}
                    except:
                        n1, n2 = 0, 0
                        params = {}
            
            else:  # One group vs population
                if outcome_type == "Dichotomous (yes/no)":
                    st.markdown("**Proportions:**")
                    p = st.number_input(
                        "Expected study proportion",
                        min_value=0.0, max_value=1.0, value=0.6, step=0.01,
                        help="Expected proportion in your study group"
                    )
                    p0 = st.number_input(
                        "Population proportion",
                        min_value=0.0, max_value=1.0, value=0.5, step=0.01,
                        help="Known proportion in the general population"
                    )
                    
                    # Calculate sample size
                    try:
                        n1 = calculate_sample_size_one_proportion(p, p0, alpha, beta, two_sided)
                        n2 = 0
                        params = {'p': p, 'p0': p0}
                    except:
                        n1, n2 = 0, 0
                        params = {}
                
                else:  # Continuous
                    st.markdown("**Means and Standard Deviation:**")
                    mu = st.number_input(
                        "Expected study mean",
                        value=105.0, step=0.1,
                        help="Expected mean in your study group"
                    )
                    mu0 = st.number_input(
                        "Population mean",
                        value=100.0, step=0.1,
                        help="Known mean in the general population"
                    )
                    sigma = st.number_input(
                        "Standard deviation",
                        min_value=0.01, value=15.0, step=0.1,
                        help="Population standard deviation"
                    )
                    
                    # Calculate sample size
                    try:
                        n1 = calculate_sample_size_one_mean(mu, mu0, sigma, alpha, beta, two_sided)
                        n2 = 0
                        params = {'mu': mu, 'mu0': mu0, 'sigma': sigma}
                    except:
                        n1, n2 = 0, 0
                        params = {}
            
            # Dropout rate adjustment
            st.markdown("**Optional Adjustments:**")
            dropout_rate = st.slider(
                "Expected dropout rate (%)",
                min_value=0, max_value=50, value=10, step=1,
                help="Percentage of subjects expected to drop out during the study"
            )
        
        with col2:
            st.markdown('<div class="section-header">Sample Size Results</div>', unsafe_allow_html=True)
            
            if n1 > 0 and n1 != float('inf'):
                # Adjust for dropout
                n1_adj = math.ceil(n1 / (1 - dropout_rate/100))
                n2_adj = math.ceil(n2 / (1 - dropout_rate/100)) if n2 > 0 else 0
                
                # Results display
                results_html = f"""
                <div class="result-box">
                    <h3 style="color: #0284c7; margin-top: 0;">📊 Required Sample Size</h3>
                """
                
                if study_type == "Two independent study groups":
                    results_html += f"""
                    <p><strong>Group 1:</strong> {n1_adj} subjects</p>
                    <p><strong>Group 2:</strong> {n2_adj} subjects</p>
                    <p><strong>Total:</strong> {n1_adj + n2_adj} subjects</p>
                    """
                else:
                    results_html += f"""
                    <p><strong>Study Group:</strong> {n1_adj} subjects</p>
                    """
                
                results_html += f"""
                    <hr style="margin: 1rem 0;">
                    <p><strong>Statistical Parameters:</strong></p>
                    <ul>
                        <li>Confidence Level: {confidence_level}% (α = {alpha})</li>
                        <li>Statistical Power: {power}% (β = {beta})</li>
                        <li>Test Type: {'Two-sided' if two_sided else 'One-sided'}</li>
                        <li>Dropout Adjustment: {dropout_rate}%</li>
                    </ul>
                </div>
                """
                
                st.markdown(results_html, unsafe_allow_html=True)
                
                # Effect size information
                if study_type == "Two independent study groups":
                    if outcome_type == "Dichotomous (yes/no)":
                        effect_size = abs(p2 - p1)
                        st.info(f"**Effect Size:** {effect_size:.3f} (absolute difference in proportions)")
                    else:
                        effect_size = abs(mu2 - mu1)
                        cohen_d = effect_size / sigma
                        st.info(f"**Effect Size:** {effect_size:.2f} units (Cohen's d = {cohen_d:.3f})")
                else:
                    if outcome_type == "Dichotomous (yes/no)":
                        effect_size = abs(p - p0)
                        st.info(f"**Effect Size:** {effect_size:.3f} (absolute difference from population)")
                    else:
                        effect_size = abs(mu - mu0)
                        cohen_d = effect_size / sigma
                        st.info(f"**Effect Size:** {effect_size:.2f} units (Cohen's d = {cohen_d:.3f})")
                
                # Generate citation
                st.markdown('<div class="section-header">📚 Citation</div>', unsafe_allow_html=True)
                citation = f"""
                <div class="citation-box">
Sample size calculated using Clinical Sample Size Calculator. 
Study design: {study_type.lower()}, {outcome_type.lower()}.
Statistical parameters: α={alpha}, β={beta}, power={power}%.
Calculated on {datetime.now().strftime('%B %d, %Y')}.
                </div>
                """
                st.markdown(citation, unsafe_allow_html=True)
                
            else:
                st.error("❌ Cannot calculate sample size with current parameters. Please check your inputs.")
    
    with tab2:
        st.markdown('<div class="section-header">📊 Mathematical Formulas</div>', unsafe_allow_html=True)
        
        # Display appropriate formula based on current selection
        if study_type == "Two independent study groups":
            if outcome_type == "Dichotomous (yes/no)":
                st.markdown('<div class="formula-box">', unsafe_allow_html=True)
                st.markdown("**Two-Sample Proportion Test Formula:**")
                st.latex(r'''
                n_1 = \frac{\left[z_{1-\alpha/2}\sqrt{\bar{p}\bar{q}\left(1+\frac{1}{r}\right)} + z_{1-\beta}\sqrt{p_1 q_1 + \frac{p_2 q_2}{r}}\right]^2}{(p_2 - p_1)^2}
                ''')
                st.markdown("Where:")
                st.markdown("""
                - $n_1$ = sample size for group 1
                - $p_1, p_2$ = expected proportions in groups 1 and 2
                - $q_1 = 1-p_1, q_2 = 1-p_2$
                - $\\bar{p} = \\frac{p_1 + rp_2}{1+r}$ (pooled proportion)
                - $r$ = allocation ratio ($n_2/n_1$)
                - $z_{1-\\alpha/2}$ = critical value for Type I error
                - $z_{1-\\beta}$ = critical value for Type II error
                """)
                st.markdown('</div>', unsafe_allow_html=True)
            
            else:  # Continuous
                st.markdown('<div class="formula-box">', unsafe_allow_html=True)
                st.markdown("**Two-Sample Mean Test Formula:**")
                st.latex(r'''
                n_1 = \frac{2\sigma^2(z_{1-\alpha/2} + z_{1-\beta})^2}{(\mu_2 - \mu_1)^2}
                ''')
                st.markdown("Where:")
                st.markdown("""
                - $n_1$ = sample size for group 1
                - $\\mu_1, \\mu_2$ = expected means in groups 1 and 2
                - $\\sigma$ = common standard deviation
                - $z_{1-\\alpha/2}$ = critical value for Type I error
                - $z_{1-\\beta}$ = critical value for Type II error
                """)
                st.markdown('</div>', unsafe_allow_html=True)
        
        else:  # One group vs population
            if outcome_type == "Dichotomous (yes/no)":
                st.markdown('<div class="formula-box">', unsafe_allow_html=True)
                st.markdown("**One-Sample Proportion Test Formula:**")
                st.latex(r'''
                n = \frac{\left[z_{1-\alpha/2}\sqrt{p_0(1-p_0)} + z_{1-\beta}\sqrt{p_1(1-p_1)}\right]^2}{(p_1 - p_0)^2}
                ''')
                st.markdown("Where:")
                st.markdown("""
                - $n$ = required sample size
                - $p_0$ = population proportion (null hypothesis)
                - $p_1$ = expected study proportion (alternative hypothesis)
                - $z_{1-\\alpha/2}$ = critical value for Type I error
                - $z_{1-\\beta}$ = critical value for Type II error
                """)
                st.markdown('</div>', unsafe_allow_html=True)
            
            else:  # Continuous
                st.markdown('<div class="formula-box">', unsafe_allow_html=True)
                st.markdown("**One-Sample Mean Test Formula:**")
                st.latex(r'''
                n = \frac{\sigma^2(z_{1-\alpha/2} + z_{1-\beta})^2}{(\mu_1 - \mu_0)^2}
                ''')
                st.markdown("Where:")
                st.markdown("""
                - $n$ = required sample size
                - $\\mu_0$ = population mean (null hypothesis)
                - $\\mu_1$ = expected study mean (alternative hypothesis)
                - $\\sigma$ = population standard deviation
                - $z_{1-\\alpha/2}$ = critical value for Type I error
                - $z_{1-\\beta}$ = critical value for Type II error
                """)
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Interactive formula parameters
        st.markdown('<div class="section-header">🎛️ Interactive Formula Visualization</div>', unsafe_allow_html=True)
        
        if 'params' in locals() and params and n1 > 0:
            # Create visualization of how parameters affect sample size
            param_ranges = {}
            if study_type == "Two independent study groups":
                if outcome_type == "Dichotomous (yes/no)":
                    param_ranges['p1'] = np.linspace(0.1, 0.9, 20)
                    param_ranges['p2'] = np.linspace(0.1, 0.9, 20)
                else:
                    param_ranges['effect_size'] = np.linspace(1, 50, 20)
                    param_ranges['sigma'] = np.linspace(5, 30, 20)
            else:
                if outcome_type == "Dichotomous (yes/no)":
                    param_ranges['effect_size'] = np.linspace(0.05, 0.5, 20)
                else:
                    param_ranges['effect_size'] = np.linspace(1, 30, 20)
            
            # Create interactive plot showing parameter sensitivity
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Effect Size vs Sample Size', 'Power vs Sample Size', 
                               'Alpha vs Sample Size', 'Parameter Sensitivity'),
                specs=[[{'secondary_y': False}, {'secondary_y': False}],
                       [{'secondary_y': False}, {'secondary_y': False}]]
            )
            
            # Effect size sensitivity
            if study_type == "Two independent study groups" and outcome_type == "Dichotomous (yes/no)":
                effects = np.linspace(0.01, 0.5, 30)
                n_sizes = []
                for eff in effects:
                    p2_test = p1 + eff
                    if 0 < p2_test < 1:
                        try:
                            n_test, _ = calculate_sample_size_two_proportions(p1, p2_test, alpha, beta, ratio, two_sided)
                            n_sizes.append(min(n_test, 2000))
                        except:
                            n_sizes.append(2000)
                    else:
                        n_sizes.append(2000)
                
                fig.add_trace(
                    go.Scatter(x=effects, y=n_sizes, name='Sample Size vs Effect Size',
                              line=dict(color='blue', width=3)),
                    row=1, col=1
                )
            
            # Power curve
            if 'params' in locals() and params:
                if study_type == "Two independent study groups" and outcome_type == "Dichotomous (yes/no)":
                    effect_range = np.linspace(0.01, 0.5, 30)
                else:
                    effect_range = np.linspace(1, 50, 30)
                
                power_values = generate_power_curve(effect_range, n1, alpha, study_type, outcome_type, **params)
                
                fig.add_trace(
                    go.Scatter(x=effect_range, y=power_values, name='Power Curve',
                              line=dict(color='red', width=3)),
                    row=1, col=2
                )
            
            fig.update_layout(height=600, showlegend=True, title_text="Parameter Sensitivity Analysis")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown('<div class="section-header">📈 Power Analysis Visualization</div>', unsafe_allow_html=True)
        
        if 'params' in locals() and params and n1 > 0:
            # Power analysis plots
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Sample size vs power
                powers = np.arange(0.5, 0.99, 0.01)
                sample_sizes = []
                
                for pwr in powers:
                    beta_test = 1 - pwr
                    try:
                        if study_type == "Two independent study groups":
                            if outcome_type == "Dichotomous (yes/no)":
                                n_test, _ = calculate_sample_size_two_proportions(
                                    params['p1'], params['p2'], alpha, beta_test, 
                                    params.get('ratio', 1), two_sided
                                )
                            else:
                                n_test, _ = calculate_sample_size_two_means(
                                    params['mu1'], params['mu2'], params['sigma'], 
                                    alpha, beta_test, params.get('ratio', 1), two_sided
                                )
                        else:
                            if outcome_type == "Dichotomous (yes/no)":
                                n_test = calculate_sample_size_one_proportion(
                                    params['p'], params['p0'], alpha, beta_test, two_sided
                                )
                            else:
                                n_test = calculate_sample_size_one_mean(
                                    params['mu'], params['mu0'], params['sigma'], 
                                    alpha, beta_test, two_sided
                                )
                        sample_sizes.append(min(n_test, 2000))
                    except:
                        sample_sizes.append(2000)
                
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(
                    x=powers * 100,
                    y=sample_sizes,
                    mode='lines',
                    name='Sample Size',
                    line=dict(color='blue', width=3)
                ))
                
                # Add current point
                fig1.add_trace(go.Scatter(
                    x=[power],
                    y=[n1],
                    mode='markers',
                    name='Current Study',
                    marker=dict(color='red', size=12, symbol='diamond')
                ))
                
                fig1.update_layout(
                    title='Sample Size vs Statistical Power',
                    xaxis_title='Statistical Power (%)',
                    yaxis_title='Required Sample Size',
                    hovermode='closest'
                )
                
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                # Effect size vs sample size
                if study_type == "Two independent study groups":
                    if outcome_type == "Dichotomous (yes/no)":
                        effects = np.linspace(0.01, min(1-params['p1'], params['p1']), 50)
                        n_values = []
                        for eff in effects:
                            try:
                                n_test, _ = calculate_sample_size_two_proportions(
                                    params['p1'], params['p1'] + eff, alpha, beta, 
                                    params.get('ratio', 1), two_sided
                                )
                                n_values.append(min(n_test, 3000))
                            except:
                                n_values.append(3000)
                        current_effect = abs(params['p2'] - params['p1'])
                    else:
                        effects = np.linspace(1, 100, 50)
                        n_values = []
                        for eff in effects:
                            try:
                                n_test, _ = calculate_sample_size_two_means(
                                    params['mu1'], params['mu1'] + eff, params['sigma'], 
                                    alpha, beta, params.get('ratio', 1), two_sided
                                )
                                n_values.append(min(n_test, 3000))
                            except:
                                n_values.append(3000)
                        current_effect = abs(params['mu2'] - params['mu1'])
                else:
                    if outcome_type == "Dichotomous (yes/no)":
                        effects = np.linspace(0.01, min(1-params['p0'], params['p0']), 50)
                        n_values = []
                        for eff in effects:
                            try:
                                n_test = calculate_sample_size_one_proportion(
                                    params['p0'] + eff, params['p0'], alpha, beta, two_sided
                                )
                                n_values.append(min(n_test, 3000))
                            except:
                                n_values.append(3000)
                        current_effect = abs(params['p'] - params['p0'])
                    else:
                        effects = np.linspace(1, 100, 50)
                        n_values = []
                        for eff in effects:
                            try:
                                n_test = calculate_sample_size_one_mean(
                                    params['mu0'] + eff, params['mu0'], params['sigma'], 
                                    alpha, beta, two_sided
                                )
                                n_values.append(min(n_test, 3000))
                            except:
                                n_values.append(3000)
                        current_effect = abs(params['mu'] - params['mu0'])
                
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=effects,
                    y=n_values,
                    mode='lines',
                    name='Sample Size',
                    line=dict(color='green', width=3)
                ))
                
                # Add current point
                fig2.add_trace(go.Scatter(
                    x=[current_effect],
                    y=[n1],
                    mode='markers',
                    name='Current Study',
                    marker=dict(color='red', size=12, symbol='diamond')
                ))
                
                fig2.update_layout(
                    title='Sample Size vs Effect Size',
                    xaxis_title='Effect Size',
                    yaxis_title='Required Sample Size',
                    hovermode='closest'
                )
                
                st.plotly_chart(fig2, use_container_width=True)
            
            # Summary statistics
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown("**📊 Power Analysis Summary:**")
            st.markdown(f"- **Current Power:** {power}%")
            st.markdown(f"- **Current Alpha:** {alpha:.3f}")
            st.markdown(f"- **Sample Size:** {n1} {'+ ' + str(n2) if n2 > 0 else ''}")
            if outcome_type == "Dichotomous (yes/no)":
                st.markdown(f"- **Effect Size:** {current_effect:.3f} (absolute difference)")
            else:
                cohen_d = current_effect / params.get('sigma', 1)
                st.markdown(f"- **Effect Size:** {current_effect:.2f} (Cohen's d = {cohen_d:.3f})")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with tab4:
        st.markdown('<div class="section-header">📚 Statistical Education</div>', unsafe_allow_html=True)
        
        # Educational content
        st.markdown("### Understanding Sample Size Calculations")
        
        st.markdown("""
        Sample size determination is crucial for designing studies with adequate statistical power 
        to detect clinically meaningful differences while controlling Type I and Type II errors.
        """)
        
        # Key concepts
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### 🎯 Key Statistical Concepts")
            
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown("""
            **Type I Error (α):**
            - Probability of rejecting true null hypothesis
            - "False positive" rate
            - Typically set at 0.05 (5%)
            
            **Type II Error (β):**
            - Probability of failing to reject false null hypothesis
            - "False negative" rate
            - Power = 1 - β
            
            **Statistical Power:**
            - Probability of detecting true effect
            - Typically set at 0.80 (80%) or higher
            """)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("#### 📏 Effect Size Guidelines")
            
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown("""
            **Cohen's d (Continuous Outcomes):**
            - Small effect: d = 0.2
            - Medium effect: d = 0.5  
            - Large effect: d = 0.8
            
            **Proportion Differences:**
            - Small effect: 0.05-0.10
            - Medium effect: 0.15-0.25
            - Large effect: 0.30+
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### ⚠️ Important Assumptions")
            
            st.markdown('<div class="warning-box">', unsafe_allow_html=True)
            st.markdown("""
            **For Valid Sample Size Calculations:**
            
            1. **Independence:** Observations must be independent
            2. **Normality:** Continuous outcomes should be normally distributed
            3. **Equal Variances:** For two-group comparisons
            4. **Random Sampling:** From target population
            5. **Fixed Parameters:** Effect size and variance estimates
            """)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("#### 🔧 Practical Considerations")
            
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown("""
            **Study Planning Tips:**
            
            - Always account for dropout/attrition
            - Consider feasibility constraints
            - Pilot studies can inform parameters
            - Consult statistician for complex designs
            - Document assumptions for protocol
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("#### 📖 References and Further Reading")
        
        st.markdown('<div class="citation-box">', unsafe_allow_html=True)
        st.markdown("""
        **Key References:**
        
        1. Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences (2nd ed.). 
           Lawrence Erlbaum Associates.
        
        2. Chow, S. C., Shao, J., Wang, H., & Lokhnygina, Y. (2017). Sample Size Calculations 
           in Clinical Research (3rd ed.). Chapman & Hall/CRC.
        
        3. Fleiss, J. L., Levin, B., & Paik, M. C. (2003). Statistical Methods for Rates 
           and Proportions (3rd ed.). John Wiley & Sons.
        
        4. Julious, S. A. (2010). Sample sizes for clinical trials. 
           Chapman & Hall/CRC.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #64748b; font-size: 0.9rem;">⚕️ Professional Clinical Research Tool • '
        'Created for Healthcare Professionals • Always Consult Statistician for Complex Studies</p>', 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
