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

# Custom CSS for professional medical styling - matching ClinCalc design
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
    
    .sample-size-table {
        background: #2c3e50;
        color: white;
    }
    
    .parameters-table {
        background: #2c3e50;
        color: white;
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
        border: 2px solid #dee2e6;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 0.5rem;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2c3e50;
    }
    
    .metric-label {
        font-size: 1rem;
        color: #6c757d;
        margin-top: 0.5rem;
    }
    
    .formula-math {
        font-family: 'Times New Roman', serif;
        font-size: 1.2rem;
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
    }
    
    .parameter-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .stSelectbox > div > div {
        background-color: white;
        border-radius: 8px;
    }
    
    .stNumberInput > div > div {
        background-color: white;
        border-radius: 8px;
    }
    
    .stSlider > div > div {
        background-color: white;
        border-radius: 8px;
    }
    
    .visualization-container {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
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
        
        # Sample size calculation using the exact ClinCalc formula
        numerator = (population_prop * q0) * (z_alpha + z_beta * math.sqrt((sample_prop * q1)/(population_prop * q0))) ** 2
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

def display_formula_with_substitution(study_design, outcome_type, params):
    """Display mathematical formulas with actual parameter substitution"""
    
    st.markdown('<div class="formula-container">', unsafe_allow_html=True)
    st.markdown("### 🧮 **Mathematical Formula & Parameter Substitution**")
    
    if study_design == "Two independent study groups":
        if outcome_type == "Continuous (means)":
            st.markdown("#### **Two-Sample T-Test Formula:**")
            
            # Display the LaTeX formula using HTML/CSS
            formula_html = """
            <div class="formula-math">
            <div style="text-align: center; font-size: 1.4rem;">
                <i>n</i> = <span style="border-top: 1px solid black; display: inline-block; padding-top: 5px;">
                    (<i>z</i><sub>1-α/2</sub> + <i>z</i><sub>1-β</sub>)<sup>2</sup> × 2σ<sup>2</sup>
                </span>
                <br>
                <span style="padding-top: 10px; display: inline-block;">(μ₁ - μ₂)<sup>2</sup></span>
            </div>
            </div>
            """
            st.markdown(formula_html, unsafe_allow_html=True)
            
            if all(key in params for key in ['mean1', 'mean2', 'std_dev', 'z_alpha', 'z_beta']):
                substitution_html = f"""
                <div class="formula-math">
                <div style="text-align: center; font-size: 1.3rem;">
                    <i>n</i> = <span style="border-top: 1px solid black; display: inline-block; padding-top: 5px;">
                        ({params['z_alpha']:.3f} + {params['z_beta']:.3f})<sup>2</sup> × 2 × ({params['std_dev']})<sup>2</sup>
                    </span>
                    <br>
                    <span style="padding-top: 10px; display: inline-block;">({params['mean1']} - {params['mean2']})<sup>2</sup></span>
                    <br><br>
                    <strong><i>n</i> = {params.get('total_per_group', 'N/A')} per group</strong>
                </div>
                </div>
                """
                st.markdown(substitution_html, unsafe_allow_html=True)
                
        else:  # Dichotomous two groups
            st.markdown("#### **Two-Proportion Z-Test Formula:**")
            
            formula_html = """
            <div class="formula-math">
            <div style="text-align: center; font-size: 1.3rem;">
                <i>n</i> = <span style="border-top: 1px solid black; display: inline-block; padding-top: 5px;">
                    [<i>z</i><sub>1-α/2</sub>√(<i>p̄q̄</i>(1 + 1/<i>k</i>)) + <i>z</i><sub>1-β</sub>√(<i>p₁q₁</i> + <i>p₂q₂</i>/<i>k</i>)]<sup>2</sup>
                </span>
                <br>
                <span style="padding-top: 10px; display: inline-block;">(<i>p₁</i> - <i>p₂</i>)<sup>2</sup></span>
            </div>
            </div>
            """
            st.markdown(formula_html, unsafe_allow_html=True)
    
    else:  # One group vs population
        if outcome_type == "Continuous (means)":
            st.markdown("#### **One-Sample T-Test Formula:**")
            
            formula_html = """
            <div class="formula-math">
            <div style="text-align: center; font-size: 1.4rem;">
                <i>N</i> = <span style="border-top: 1px solid black; display: inline-block; padding-top: 5px;">
                    (<i>z</i><sub>1-α/2</sub> + <i>z</i><sub>1-β</sub>)<sup>2</sup> × σ<sup>2</sup>
                </span>
                <br>
                <span style="padding-top: 10px; display: inline-block;">(μ<sub>sample</sub> - μ<sub>population</sub>)<sup>2</sup></span>
            </div>
            </div>
            """
            st.markdown(formula_html, unsafe_allow_html=True)
            
        else:  # Dichotomous one group  
            st.markdown("#### **One-Sample Proportion Test Formula:**")
            
            # Display the exact ClinCalc formula
            formula_html = """
            <div class="formula-math">
            <div style="text-align: center; font-size: 1.3rem;">
                <i>N</i> = <span style="border-top: 1px solid black; display: inline-block; padding-top: 5px;">
                    <i>p₀q₀</i> {<i>z</i><sub>1-α/2</sub> + <i>z</i><sub>1-β</sub> √(<i>p₁q₁</i>/<i>p₀q₀</i>)}<sup>2</sup>
                </span>
                <br>
                <span style="padding-top: 10px; display: inline-block;">(<i>p₁</i> - <i>p₀</i>)<sup>2</sup></span>
            </div>
            </div>
            """
            st.markdown(formula_html, unsafe_allow_html=True)
            
            # Show parameter substitution if available
            if all(key in params for key in ['p0', 'p1', 'q0', 'q1', 'z_alpha', 'z_beta']):
                substitution_html = f"""
                <div class="formula-math">
                <div style="text-align: center; font-size: 1.2rem;">
                    <p><strong>Parameter Substitution:</strong></p>
                    <i>p₀</i> = {params['p0']} (population proportion)<br>
                    <i>p₁</i> = {params['p1']} (study proportion)<br>
                    <i>q₀</i> = 1 - <i>p₀</i> = {params['q0']}<br>
                    <i>q₁</i> = 1 - <i>p₁</i> = {params['q1']}<br><br>
                    
                    <i>N</i> = <span style="border-top: 1px solid black; display: inline-block; padding-top: 5px;">
                        {params['p0']} × {params['q0']} × {{{params['z_alpha']:.3f} + {params['z_beta']:.3f} × √({params['p1']} × {params['q1']}/{params['p0']} × {params['q0']})}}<sup>2</sup>
                    </span>
                    <br>
                    <span style="padding-top: 10px; display: inline-block;">({params['p1']} - {params['p0']})<sup>2</sup></span>
                    <br><br>
                    <strong><i>N</i> = {params.get('n', 'N/A')}</strong>
                </div>
                </div>
                """
                st.markdown(substitution_html, unsafe_allow_html=True)
    
    # Parameter definitions
    st.markdown("#### **Parameter Definitions:**")
    col1, col2 = st.columns(2)
    with col1:
        if outcome_type == "Continuous (means)":
            st.markdown("""
            - **μ** = mean values
            - **σ** = standard deviation
            - **n** = sample size per group
            """)
        else:
            st.markdown("""
            - **p** = proportion/probability
            - **q** = 1 - p (complement)
            - **N** = total sample size
            """)
    
    with col2:
        st.markdown(f"""
        - **α** = {params.get('alpha', 0.05)} (Type I error)
        - **β** = {1 - params.get('power', 0.80):.2f} (Type II error)
        - **Power** = {params.get('power', 0.80)} (1-β)
        - **z₁₋α/₂** = {params.get('z_alpha', 'N/A'):.3f}
        - **z₁₋β** = {params.get('z_beta', 'N/A'):.3f}
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)

def create_professional_visualizations(study_design, outcome_type, base_params, results):
    """Create professional parameter sensitivity analysis charts"""
    
    st.markdown('<div class="visualization-container">', unsafe_allow_html=True)
    st.markdown("### 🔬 **Interactive Formula Visualization**")
    st.markdown("#### **Parameter Sensitivity Analysis**")
    
    # Create 2x2 subplot layout
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Effect Size vs Sample Size', 'Power vs Sample Size', 
                       'Alpha vs Sample Size', 'Parameter Sensitivity'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    if study_design == "Two independent study groups" and outcome_type == "Continuous (means)":
        # Effect Size vs Sample Size
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
        
        fig.add_trace(go.Scatter(x=effect_sizes, y=sample_sizes, mode='lines+markers', 
                                name='Effect Size', line=dict(color='#2E86AB', width=3)), 
                      row=1, col=1)
        
        # Power vs Sample Size
        powers = np.linspace(0.70, 0.95, 20)
        power_sample_sizes = []
        
        for power in powers:
            try:
                result = SampleSizeCalculator.calculate_continuous_two_groups(
                    base_params['mean1'], base_params['mean2'],
                    base_params['std_dev'], base_params['alpha'], power
                )
                power_sample_sizes.append(result['total'])
            except:
                power_sample_sizes.append(np.nan)
        
        fig.add_trace(go.Scatter(x=powers, y=power_sample_sizes, mode='lines+markers',
                                name='Power Curve', line=dict(color='#A23B72', width=3)),
                      row=1, col=2)
    
    elif study_design == "One study group vs. population" and outcome_type == "Dichotomous (yes/no)":
        # Effect Size vs Sample Size for proportions
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
        
        fig.add_trace(go.Scatter(x=effect_sizes, y=sample_sizes, mode='lines+markers',
                                name='Effect Size', line=dict(color='#2E86AB', width=3)),
                      row=1, col=1)
        
        # Power vs Sample Size
        powers = np.linspace(0.70, 0.95, 20)
        power_sample_sizes = []
        
        for power in powers:
            try:
                result = SampleSizeCalculator.calculate_proportions_one_group(
                    base_params['sample_prop'], base_params['population_prop'],
                    base_params['alpha'], power
                )
                power_sample_sizes.append(result['n'])
            except:
                power_sample_sizes.append(np.nan)
        
        fig.add_trace(go.Scatter(x=powers, y=power_sample_sizes, mode='lines+markers',
                                name='Power Curve', line=dict(color='#A23B72', width=3)),
                      row=1, col=2)
    
    # Alpha vs Sample Size
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
    
    fig.add_trace(go.Bar(x=alphas, y=alpha_sample_sizes, name='Alpha Levels',
                        marker_color='#28a745'), row=2, col=1)
    
    # Parameter Sensitivity - show current parameters as bar chart
    param_names = ['Effect Size', 'Power', 'Alpha']
    param_values = [
        results.get('effect_size', 0) * 100,  # Convert to percentage scale
        base_params.get('power', 0.8) * 100,
        base_params.get('alpha', 0.05) * 100
    ]
    
    fig.add_trace(go.Bar(x=param_names, y=param_values, name='Current Parameters',
                        marker_color=['#2E86AB', '#A23B72', '#28a745']), row=2, col=2)
    
    # Update layout
    fig.update_layout(
        height=800,
        showlegend=False,
        title_text="Parameter Sensitivity Analysis",
        title_x=0.5,
        font=dict(size=12)
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Effect Size (Cohen's d)", row=1, col=1)
    fig.update_yaxes(title_text="Sample Size", row=1, col=1)
    fig.update_xaxes(title_text="Statistical Power", row=1, col=2)
    fig.update_yaxes(title_text="Sample Size", row=1, col=2)
    fig.update_xaxes(title_text="Alpha Level", row=2, col=1)
    fig.update_yaxes(title_text="Sample Size", row=2, col=1)
    fig.update_xaxes(title_text="Parameters", row=2, col=2)
    fig.update_yaxes(title_text="Values (%)", row=2, col=2)
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

def display_professional_results(results, study_design, outcome_type, params):
    """Display results in ClinCalc professional format"""
    
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
    
    # Results tables
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### **Sample Size**")
        if study_design == "Two independent study groups":
            sample_data = {
                "Group": ["Group 1", "Group 2", "Total"],
                "Size": [results['n1'], results['n2'], results['total']]
            }
        else:
            sample_data = {
                "Group": ["Group 1", "Total"],
                "Size": [results['n'], results['n']]
            }
        
        sample_df = pd.DataFrame(sample_data)
        
        # Create HTML table with styling
        table_html = """
        <table style="width:100%; border-collapse: collapse; margin: 1rem 0;">
            <thead style="background-color: #2c3e50; color: white;">
                <tr>
                    <th style="padding: 12px; text-align: left; border: 1px solid #ddd;">Group</th>
                    <th style="padding: 12px; text-align: center; border: 1px solid #ddd;">Size</th>
                </tr>
            </thead>
            <tbody>
        """
        
        for i, row in sample_df.iterrows():
            bg_color = "#f8f9fa" if i % 2 == 0 else "white"
            if row['Group'] == 'Total':
                bg_color = "#e9ecef"
                weight = "font-weight: bold;"
            else:
                weight = ""
            
            table_html += f"""
                <tr style="background-color: {bg_color};">
                    <td style="padding: 10px; border: 1px solid #ddd; {weight}">{row['Group']}</td>
                    <td style="padding: 10px; text-align: center; border: 1px solid #ddd; {weight}">{row['Size']}</td>
                </tr>
            """
        
        table_html += "</tbody></table>"
        st.markdown(table_html, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### **Study Parameters**")
        
        # Build parameters data based on study type
        if study_design == "Two independent study groups":
            if outcome_type == "Continuous (means)":
                param_data = {
                    "Parameter": ["Mean, group 1", "Mean, group 2", "Standard deviation", "Alpha", "Beta", "Power"],
                    "Value": [params['mean1'], params['mean2'], params['std_dev'], 
                             params['alpha'], 1-params['power'], params['power']]
                }
            else:
                param_data = {
                    "Parameter": ["Proportion, group 1", "Proportion, group 2", "Alpha", "Beta", "Power"],
                    "Value": [f"{params['p1']:.1%}", f"{params['p2']:.1%}", 
                             params['alpha'], 1-params['power'], params['power']]
                }
        else:
            if outcome_type == "Continuous (means)":
                param_data = {
                    "Parameter": ["Mean, sample", "Mean, population", "Standard deviation", "Alpha", "Beta", "Power"],
                    "Value": [params['sample_mean'], params['population_mean'], params['std_dev'],
                             params['alpha'], 1-params['power'], params['power']]
                }
            else:
                param_data = {
                    "Parameter": ["Incidence, population", "Incidence, study group", "Alpha", "Beta", "Power"],
                    "Value": [f"{params['population_prop']:.0%}", f"{params['sample_prop']:.0%}",
                             params['alpha'], 1-params['power'], params['power']]
                }
        
        param_df = pd.DataFrame(param_data)
        
        # Create parameters table
        param_table_html = """
        <table style="width:100%; border-collapse: collapse; margin: 1rem 0;">
            <thead style="background-color: #2c3e50; color: white;">
                <tr>
                    <th style="padding: 12px; text-align: left; border: 1px solid #ddd;">Parameter</th>
                    <th style="padding: 12px; text-align: center; border: 1px solid #ddd;">Value</th>
                </tr>
            </thead>
            <tbody>
        """
        
        for i, row in param_df.iterrows():
            bg_color = "#f8f9fa" if i % 2 == 0 else "white"
            param_table_html += f"""
                <tr style="background-color: {bg_color};">
                    <td style="padding: 10px; border: 1px solid #ddd;">{row['Parameter']}</td>
                    <td style="padding: 10px; text-align: center; border: 1px solid #ddd;">{row['Value']}</td>
                </tr>
            """
        
        param_table_html += "</tbody></table>"
        st.markdown(param_table_html, unsafe_allow_html=True)

def generate_professional_citation(study_design, outcome_type, params, results):
    """Generate comprehensive AMA-style citation"""
    current_date = datetime.now().strftime("%B %d, %Y")
    
    # Determine study description
    if study_design == "Two independent study groups":
        study_desc = "two independent study groups"
    else:
        study_desc = "one study group vs. population"
    
    if outcome_type == "Continuous (means)":
        outcome_desc = "continuous"
    else:
        outcome_desc = "dichotomous (yes/no)"
    
    # Build parameter string
    param_str = f"α={params.get('alpha', 0.05)}, β={1-params.get('power', 0.80):.1f}, power={params.get('power', 0.80)*100:.0f}%"
    
    citation = f"""Sample size calculated using Clinical Sample Size Calculator. Study design: {study_desc}, {outcome_desc}. Statistical parameters: {param_str}. Calculated on {current_date}."""
    
    return citation

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🧮 ClinCalc Sample Size Calculator</h1>
        <p>Professional Statistical Power Analysis for Clinical Research</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
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
            st.markdown("### Expected dropout rate (%)")
            dropout_rate = st.slider(
                "",
                min_value=0,
                max_value=50,
                value=10,
                help="Expected dropout percentage"
            ) / 100
        else:
            dropout_rate = 0.0
    
    # Main content area - single column for professional layout
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
            
            if st.button("🔢 **Calculate Sample Size**", type="primary"):
                try:
                    results = SampleSizeCalculator.calculate_continuous_two_groups(
                        mean1, mean2, std_dev, alpha, power, allocation_ratio, two_sided, dropout_rate
                    )
                    
                    params = {
                        'mean1': mean1, 'mean2': mean2, 'std_dev': std_dev,
                        'alpha': alpha, 'power': power, 'z_alpha': results['z_alpha'],
                        'z_beta': results['z_beta'], 'allocation_ratio': allocation_ratio,
                        'total_per_group': results['n1']
                    }
                    
                    display_formula_with_substitution(study_design, outcome_type, params)
                    display_professional_results(results, study_design, outcome_type, params)
                    create_professional_visualizations(study_design, outcome_type, params, results)
                    
                    # Citation
                    st.markdown("### 📋 **Citation**")
                    citation = generate_professional_citation(study_design, outcome_type, params, results)
                    st.markdown(f'<div class="citation-container">{citation}</div>', unsafe_allow_html=True)
                    
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
            
            if st.button("🔢 **Calculate Sample Size**", type="primary"):
                try:
                    results = SampleSizeCalculator.calculate_proportions_two_groups(
                        p1, p2, alpha, power, allocation_ratio, two_sided, True, dropout_rate
                    )
                    
                    params = {
                        'p1': p1, 'p2': p2, 'alpha': alpha, 'power': power,
                        'z_alpha': results['z_alpha'], 'z_beta': results['z_beta'],
                        'p_pooled': results['p_pooled'], 'allocation_ratio': allocation_ratio
                    }
                    
                    display_formula_with_substitution(study_design, outcome_type, params)
                    display_professional_results(results, study_design, outcome_type, params)
                    
                    # Citation
                    st.markdown("### 📋 **Citation**")
                    citation = generate_professional_citation(study_design, outcome_type, params, results)
                    st.markdown(f'<div class="citation-container">{citation}</div>', unsafe_allow_html=True)
                    
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
            
            if st.button("🔢 **Calculate Sample Size**", type="primary"):
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
                    
                    display_formula_with_substitution(study_design, outcome_type, params)
                    display_professional_results(results, study_design, outcome_type, params)
                    
                    # Citation
                    st.markdown("### 📋 **Citation**")
                    citation = generate_professional_citation(study_design, outcome_type, params, results)
                    st.markdown(f'<div class="citation-container">{citation}</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Calculation error: {str(e)}")
        
        else:  # Dichotomous outcomes
            st.markdown("### **Proportions:**")
            col_a, col_b = st.columns(2)
            with col_a:
                sample_prop = st.slider("**Expected study proportion**", 0.01, 0.99, 0.50, 0.01)
            with col_b:
                population_prop = st.slider("**Population proportion**", 0.01, 0.99, 0.40, 0.01)
            
            if st.button("🔢 **Calculate Sample Size**", type="primary"):
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
                    
                    display_formula_with_substitution(study_design, outcome_type, params)
                    display_professional_results(results, study_design, outcome_type, params)
                    create_professional_visualizations(study_design, outcome_type, params, results)
                    
                    # Effect size display
                    st.markdown(f"**Effect Size:** {results['effect_size']:.3f} (absolute difference from population)")
                    
                    # Citation
                    st.markdown("### 📋 **Citation**")
                    citation = generate_professional_citation(study_design, outcome_type, params, results)
                    st.markdown(f'<div class="citation-container">{citation}</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Calculation error: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6c757d; font-size: 0.9rem;">
        🏥 Professional Clinical Research Tool • Created for Healthcare Professionals • Always Consult Statistician for Complex Studies
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
