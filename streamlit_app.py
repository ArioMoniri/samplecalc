chimport streamlit as st
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
    
    .citation-container {
        background: #fff3cd;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #ffc107;
        margin: 1.5rem 0;
    }
    
    .info-box {
        background: linear-gradient(145deg, #e3f2fd, #bbdefb);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(33,150,243,0.2);
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
    
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.2rem;
        font-weight: bold;
    }
    
    div[data-testid="metric-container"] {
        background-color: white;
        border: 2px solid #28a745;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(40,167,69,0.2);
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
        
        # Continuity correction
        if continuity_correction:
            correction_factor = 1 + math.sqrt(1 + 4/abs(p1-p2)*math.sqrt(n1))
            n1 = n1 * (correction_factor ** 2) / 4
        
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
        
        # Continuity correction
        if continuity_correction:
            correction = 1 + math.sqrt(1 + 4/abs(sample_prop - population_prop)*math.sqrt(n))
            n = n * (correction ** 2) / 4
        
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

def display_latex_formula(study_design, outcome_type, params):
    """Display proper LaTeX formulas using Streamlit's latex function"""
    
    st.markdown('<div class="formula-container">', unsafe_allow_html=True)
    st.markdown("### 🧮 **Mathematical Formula & Parameter Substitution**")
    
    if study_design == "Two independent study groups":
        if outcome_type == "Continuous (means)":
            st.markdown("#### **Two-Sample T-Test Formula:**")
            st.latex(r'''n = \frac{(z_{1-\alpha/2} + z_{1-\beta})^2 \cdot 2\sigma^2}{(\mu_1 - \mu_2)^2}''')
            
            if all(key in params for key in ['mean1', 'mean2', 'std_dev', 'z_alpha', 'z_beta']):
                st.markdown("#### **Parameter Substitution:**")
                st.latex(f'''n = \\frac{{({params['z_alpha']:.3f} + {params['z_beta']:.3f})^2 \\cdot 2 \\cdot ({params['std_dev']})^2}}{{({params['mean1']} - {params['mean2']})^2}}''')
                
        else:  # Dichotomous two groups
            st.markdown("#### **Two-Proportion Z-Test Formula:**")
            st.latex(r'''n = \frac{[z_{1-\alpha/2}\sqrt{\bar{p}\bar{q}(1 + \frac{1}{k})} + z_{1-\beta}\sqrt{p_1q_1 + \frac{p_2q_2}{k}}]^2}{(p_1 - p_2)^2}''')
    
    else:  # One group vs population
        if outcome_type == "Continuous (means)":
            st.markdown("#### **One-Sample T-Test Formula:**")
            st.latex(r'''N = \frac{(z_{1-\alpha/2} + z_{1-\beta})^2 \cdot \sigma^2}{(\mu_{sample} - \mu_{population})^2}''')
            
        else:  # Dichotomous one group  
            st.markdown("#### **One-Sample Proportion Test Formula:**")
            st.latex(r'''N = \frac{p_0q_0 \{z_{1-\alpha/2} + z_{1-\beta} \sqrt{\frac{p_1q_1}{p_0q_0}}\}^2}{(p_1 - p_0)^2}''')
            
            # Show parameter substitution if available
            if all(key in params for key in ['p0', 'p1', 'q0', 'q1', 'z_alpha', 'z_beta']):
                st.markdown("#### **Parameter Substitution:**")
                st.markdown(f"""
                - p₀ = {params['p0']} (population proportion)
                - p₁ = {params['p1']} (study proportion)
                - q₀ = 1 - p₀ = {params['q0']:.3f}
                - q₁ = 1 - p₁ = {params['q1']:.3f}
                """)
                
                st.latex(f'''N = \\frac{{{params['p0']} \\times {params['q0']:.3f} \\times \\{{1.960 + 0.842 \\times \\sqrt{{\\frac{{{params['p1']} \\times {params['q1']:.3f}}}{{{params['p0']} \\times {params['q0']:.3f}}}}}\\}}^2}}{{({params['p1']} - {params['p0']})^2}} = {params.get('n', 'N/A')}''')
    
    st.markdown('</div>', unsafe_allow_html=True)

def create_enhanced_visualizations(study_design, outcome_type, base_params, results):
    """Create enhanced parameter sensitivity analysis charts"""
    
    st.markdown('<div class="visualization-container">', unsafe_allow_html=True)
    st.markdown("### 📊 **Interactive Parameter Sensitivity Analysis**")
    
    # Create tabs for different visualization types
    tab1, tab2, tab3 = st.tabs(["📈 Sensitivity Curves", "🎯 Power Analysis", "📊 Comparison Charts"])
    
    with tab1:
        # Effect Size vs Sample Size
        fig1 = go.Figure()
        
        if study_design == "Two independent study groups" and outcome_type == "Continuous (means)":
            effect_sizes = np.linspace(0.2, 2.0, 30)
            sample_sizes = []
            
            for es in effect_sizes:
                mean_diff = es * base_params['std_dev']
                try:
                    result = SampleSizeCalculator.calculate_continuous_two_groups(
                        base_params['mean1'], 
                        base_params['mean1'] + mean_diff,
                        base_params['std_dev'],
                        base_params['alpha'],
                        base_params['power']
                    )
                    sample_sizes.append(result['total'])
                except:
                    sample_sizes.append(np.nan)
            
            fig1.add_trace(go.Scatter(
                x=effect_sizes, y=sample_sizes, 
                mode='lines+markers',
                name='Effect Size vs Sample Size',
                line=dict(color='#2E86AB', width=4),
                marker=dict(size=8)
            ))
            
            fig1.update_layout(
                title="Effect Size vs Required Sample Size",
                xaxis_title="Effect Size (Cohen's d)",
                yaxis_title="Total Sample Size Required",
                height=500,
                showlegend=True,
                font=dict(size=14)
            )
            
        elif study_design == "One study group vs. population" and outcome_type == "Dichotomous (yes/no)":
            effect_sizes = np.linspace(0.05, 0.40, 30)
            sample_sizes = []
            
            for es in effect_sizes:
                try:
                    result = SampleSizeCalculator.calculate_proportions_one_group(
                        base_params['population_prop'] + es,
                        base_params['population_prop'],
                        base_params['alpha'],
                        base_params['power']
                    )
                    sample_sizes.append(result['n'])
                except:
                    sample_sizes.append(np.nan)
            
            fig1.add_trace(go.Scatter(
                x=effect_sizes, y=sample_sizes,
                mode='lines+markers',
                name='Effect Size vs Sample Size',
                line=dict(color='#2E86AB', width=4),
                marker=dict(size=8)
            ))
            
            fig1.update_layout(
                title="Effect Size vs Required Sample Size",
                xaxis_title="Effect Size (Absolute Difference in Proportions)",
                yaxis_title="Sample Size Required",
                height=500,
                showlegend=True,
                font=dict(size=14)
            )
        
        st.plotly_chart(fig1, use_container_width=True)
    
    with tab2:
        # Power Analysis
        powers = np.linspace(0.70, 0.95, 20)
        power_sample_sizes = []
        
        for power in powers:
            try:
                if study_design == "Two independent study groups":
                    if outcome_type == "Continuous (means)":
                        result = SampleSizeCalculator.calculate_continuous_two_groups(
                            base_params['mean1'], base_params['mean2'],
                            base_params['std_dev'], base_params['alpha'], power
                        )
                        power_sample_sizes.append(result['total'])
                else:
                    if outcome_type == "Dichotomous (yes/no)":
                        result = SampleSizeCalculator.calculate_proportions_one_group(
                            base_params['sample_prop'], base_params['population_prop'],
                            base_params['alpha'], power
                        )
                        power_sample_sizes.append(result['n'])
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
        
        for alpha in alphas:
            try:
                if study_design == "Two independent study groups":
                    if outcome_type == "Continuous (means)":
                        result = SampleSizeCalculator.calculate_continuous_two_groups(
                            base_params['mean1'], base_params['mean2'],
                            base_params['std_dev'], alpha, base_params['power']
                        )
                        alpha_sample_sizes.append(result['total'])
                else:
                    if outcome_type == "Dichotomous (yes/no)":
                        result = SampleSizeCalculator.calculate_proportions_one_group(
                            base_params['sample_prop'], base_params['population_prop'],
                            alpha, base_params['power']
                        )
                        alpha_sample_sizes.append(result['n'])
            except:
                alpha_sample_sizes.append(np.nan)
        
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=['α = 0.01', 'α = 0.05', 'α = 0.10'],
            y=alpha_sample_sizes,
            name='Alpha Levels',
            marker_color=['#e74c3c', '#f39c12', '#27ae60'],
            text=alpha_sample_sizes,
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
    
    st.markdown('</div>', unsafe_allow_html=True)

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

def generate_multiple_citations(study_design, outcome_type, params, results):
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
    
    # Main layout with sidebar
    with st.sidebar:
        st.markdown("## 📊 Study Configuration")
        
        st.markdown("**Select Study Design:**")
        study_design = st.radio(
            "",
            ["Two independent study groups", "One study group vs. population"],
            help="Choose your study design type"
        )
        
        st.markdown("**Select Outcome Type:**")
        outcome_type = st.radio(
            "",
            ["Continuous (means)", "Dichotomous (yes/no)"],
            help="Select the type of your primary outcome variable"
        )
        
        st.markdown("---")
        st.markdown("### ⚙️ Statistical Parameters")
        
        confidence_level = st.selectbox(
            "**Confidence Level (%)**",
            [90, 95, 99],
            index=1,
            help="Confidence level for the statistical test"
        )
        alpha = (100 - confidence_level) / 100
        
        power_percent = st.selectbox(
            "**Statistical Power (%)**",
            [70, 80, 90, 95],
            index=1,
            help="Statistical power (1-β)"
        )
        power = power_percent / 100
        
        two_sided = st.checkbox(
            "**Two-sided test**",
            value=True,
            help="Use two-sided hypothesis test"
        )
        
        if outcome_type == "Dichotomous (yes/no)":
            st.markdown("**Expected dropout rate (%)**")
            dropout_rate = st.slider(
                "",
                min_value=0,
                max_value=50,
                value=10,
                help="Expected dropout percentage"
            ) / 100
        else:
            dropout_rate = 0.0
    
    # Main content area with tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Calculator", "🧮 Formula", "📊 Analysis", "📄 Citation"])
    
    with tab1:
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
                col_a, col_b = st.columns(2)
                with col_a:
                    p1 = st.slider("**Group 1 proportion**", 0.01, 0.99, 0.30, 0.01)
                with col_b:
                    p2 = st.slider("**Group 2 proportion**", 0.01, 0.99, 0.50, 0.01)
                
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
                col_a, col_b = st.columns(2)
                with col_a:
                    sample_prop = st.slider("**Expected study proportion**", 0.01, 0.99, 0.33, 0.01)
                with col_b:
                    population_prop = st.slider("**Population proportion**", 0.01, 0.99, 0.40, 0.01)
                
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
    
    with tab2:
        if 'params' in st.session_state:
            display_latex_formula(
                st.session_state.study_design,
                st.session_state.outcome_type,
                st.session_state.params
            )
        else:
            st.info("Please calculate sample size first to view the formula.")
    
    with tab3:
        if 'results' in st.session_state:
            create_enhanced_visualizations(
                st.session_state.study_design,
                st.session_state.outcome_type,
                st.session_state.params,
                st.session_state.results
            )
        else:
            st.info("Please calculate sample size first to view the analysis.")
    
    with tab4:
        if 'results' in st.session_state:
            st.markdown("### 📄 **Citation Formats**")
            
            citations = generate_multiple_citations(
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
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6c757d; font-size: 0.9rem;">
        🏥 Professional Clinical Research Tool • Created for Healthcare Professionals • Always Consult Statistician for Complex Studies
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
