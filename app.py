import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Wine Quality Predictor",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
# Custom CSS styling
st.markdown("""
    <style>
    html, body, [class*="css"]  {
        font-family: 'Segoe UI', sans-serif;
    }
    .main-header {
        font-size: 3.5rem;
        font-weight: bold;
        color: #800000; /* deep maroon */
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #a0522d; /* sienna */
        text-align: center;
        margin-bottom: 2rem;
        font-style: italic;
    }
    .prediction-card {
        background: linear-gradient(135deg, #800000, #b22222); /* maroon to firebrick */
        color: white;
        padding: 2.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.25);
    }
    .feature-card {
        background: #f2f2f2;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #800000;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        border: 1px solid #dcdcdc;
    }
    .metric-card h4 {
        font-size: 1.1rem;
        color: #800000;
        margin-bottom: 0.5rem;
    }
    .metric-card h2 {
        font-size: 1.8rem;
        color: #333333;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🍷 Wine Quality Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced AI-powered wine quality assessment</p>', unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None

# Robust model loading
def load_model_robust():
    """Load model with comprehensive error handling"""
    try:
        with open('model.pkl', 'rb') as file:
            model_data = pickle.load(file)
        
        st.success("✅ Model loaded successfully!")
        return model_data
        
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        st.info("Please ensure model.pkl exists and is properly formatted")
        return None

# Load model
model_data = load_model_robust()

# Extract model components
model = None
scaler = None
feature_names = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
    'pH', 'sulphates', 'alcohol'
]

if model_data is not None:
    try:
        if isinstance(model_data, dict):
            # Handle dictionary format
            model = model_data.get('model') or model_data.get('regressor')
            scaler = model_data.get('scaler')
            feature_names = model_data.get('feature_names', feature_names)
        else:
            # Handle direct model format
            model = model_data
            
        if model is None:
            st.error("❌ No valid model found in the file")
            
    except Exception as e:
        st.error(f"❌ Error processing model data: {e}")

# Sidebar navigation
st.sidebar.header("🧭 Navigation")
page = st.sidebar.selectbox("Choose a section:", 
    ["🎯 Predict Quality", "📊 Data Analysis", "🤖 Model Info", "📖 Usage Guide"])

# Check if model is ready
if model is None:
    st.warning("⚠️ Model not available. Please check model.pkl file.")
    st.stop()

if page == "🎯 Predict Quality":
    st.header("🔮 Wine Quality Prediction")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🍇 Enter Wine Characteristics")
        
        col_input1, col_input2 = st.columns(2)
        
        with col_input1:
            fixed_acidity = st.slider("Fixed Acidity", 4.0, 16.0, 8.3, 0.1,
                                    help="Total acid content")
            volatile_acidity = st.slider("Volatile Acidity", 0.1, 1.6, 0.53, 0.01,
                                       help="Acetic acid content")
            citric_acid = st.slider("Citric Acid", 0.0, 1.0, 0.27, 0.01,
                                  help="Citric acid content")
            residual_sugar = st.slider("Residual Sugar", 0.9, 15.0, 2.5, 0.1,
                                     help="Remaining sugar after fermentation")
            chlorides = st.slider("Chlorides", 0.01, 0.6, 0.09, 0.01,
                                help="Sodium chloride content")
            free_sulfur_dioxide = st.slider("Free SO₂", 1.0, 72.0, 15.0, 1.0,
                                          help="Free sulfur dioxide")
        
        with col_input2:
            total_sulfur_dioxide = st.slider("Total SO₂", 6.0, 289.0, 46.0, 1.0,
                                           help="Total sulfur dioxide")
            density = st.slider("Density", 0.990, 1.040, 0.996, 0.001,
                              help="Wine density")
            ph = st.slider("pH Level", 2.7, 4.0, 3.3, 0.1,
                         help="Acidity level")
            sulphates = st.slider("Sulphates", 0.3, 2.0, 0.66, 0.01,
                                help="Potassium sulfate content")
            alcohol = st.slider("Alcohol %", 8.0, 15.0, 10.4, 0.1,
                              help="Alcohol content")
        
        # Create prediction button
        if st.button("🎯 Predict Quality", type="primary", use_container_width=True):
            # Prepare input
            input_features = np.array([[fixed_acidity, volatile_acidity, citric_acid, 
                                      residual_sugar, chlorides, free_sulfur_dioxide,
                                      total_sulfur_dioxide, density, ph, sulphates, alcohol]])
            
            try:
                # Make prediction based on model type
                try:
                    # Try direct prediction
                    prediction = float(model.predict(input_features)[0])
                    
                    # Ensure prediction is within reasonable bounds
                    prediction = max(3, min(8, prediction))
                    
                except Exception as model_error:
                    # Handle specific model errors
                    st.error(f"Model prediction error: {model_error}")
                    prediction = 5.5  # Safe fallback
                
                # Display results
                st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                st.markdown(f"<h2>🎯 Predicted Quality: {prediction:.1f}/10</h2>", 
                           unsafe_allow_html=True)
                
                # Quality assessment
                if prediction >= 7:
                    st.success("🎉 **Excellent Quality** - Premium wine characteristics")
                elif prediction >= 6:
                    st.info("👍 **Good Quality** - Well-balanced wine")
                elif prediction >= 5:
                    st.warning("⚖️ **Average Quality** - Acceptable wine")
                else:
                    st.error("📉 **Below Average** - Needs improvement")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Feature analysis
                st.subheader("📊 Feature Analysis")
                
                # Comparison with averages
                avg_values = [8.32, 0.53, 0.27, 2.54, 0.09, 15.87, 46.47, 0.996, 3.31, 0.66, 10.42]
                features = ['Fixed Acidity', 'Volatile Acidity', 'Citric Acid', 'Residual Sugar',
                           'Chlorides', 'Free SO₂', 'Total SO₂', 'Density', 'pH', 'Sulphates', 'Alcohol']
                
                fig, ax = plt.subplots(figsize=(12, 6))
                x = np.arange(len(features))
                width = 0.35
                
                bars1 = ax.bar(x - width/2, input_features[0], width, 
                             label='Your Wine', color='#8B0000', alpha=0.8)
                bars2 = ax.bar(x + width/2, avg_values, width, 
                             label='Average', color='#722f37', alpha=0.6)
                
                ax.set_xlabel('Features')
                ax.set_ylabel('Values')
                ax.set_title('Wine Characteristics Comparison')
                ax.set_xticks(x)
                ax.set_xticklabels(features, rotation=45, ha='right')
                ax.legend()
                plt.tight_layout()
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Prediction error: {e}")
                st.info("Please check your model format and ensure it's properly trained")
    
    with col2:
        st.subheader("📊 Current Values")
        
        # Display current input values in cards
        metrics = [
            ("Fixed Acidity", fixed_acidity, "g/L"),
            ("Volatile Acidity", volatile_acidity, "g/L"),
            ("Citric Acid", citric_acid, "g/L"),
            ("Residual Sugar", residual_sugar, "g/L"),
            ("Chlorides", chlorides, "g/L"),
            ("Free SO₂", free_sulfur_dioxide, "mg/L"),
            ("Total SO₂", total_sulfur_dioxide, "mg/L"),
            ("Density", density, "g/cm³"),
            ("pH", ph, ""),
            ("Sulphates", sulphates, "g/L"),
            ("Alcohol", alcohol, "%")
        ]
        
        for name, value, unit in metrics:
            st.markdown(f"<div class='metric-card'><h4>{name}</h4><h2>{value:.2f} {unit}</h2></div>", 
                       unsafe_allow_html=True)

elif page == "📊 Data Analysis":
    st.header("📊 Comprehensive Data Analysis")
    
    try:
        df = pd.read_csv('winequality-red.csv')
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"<div class='metric-card'><h3>📊 Total Records</h3><h2>{len(df):,}</h2></div>", 
                       unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='metric-card'><h3>🎯 Features</h3><h2>{len(df.columns)-1}</h2></div>", 
                       unsafe_allow_html=True)
        with col3:
            st.markdown(f"<div class='metric-card'><h3>⭐ Quality Range</h3><h2>{df['quality'].min()}-{df['quality'].max()}</h2></div>", 
                       unsafe_allow_html=True)
        with col4:
            st.markdown(f"<div class='metric-card'><h3>📈 Avg Quality</h3><h2>{df['quality'].mean():.1f}</h2></div>", 
                       unsafe_allow_html=True)
        
        # Create tabs for different analyses
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Quality Distribution", "🔗 Correlations", "📋 Statistics", "🔍 Sample Data"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(figsize=(10, 6))
                quality_counts = df['quality'].value_counts().sort_index()
                bars = ax.bar(quality_counts.index, quality_counts.values, 
                            color=['#8B0000', '#A52A2A', '#CD5C5C', '#DC143C', '#B22222', '#FF0000'])
                ax.set_xlabel('Wine Quality Rating')
                ax.set_ylabel('Number of Wines')
                ax.set_title('Distribution of Wine Quality Ratings')
                ax.set_xticks(quality_counts.index)
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}', ha='center', va='bottom')
                
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                st.markdown("### 📊 Quality Insights")
                st.write(f"**Most Common Quality:** {quality_counts.idxmax()} ({quality_counts.max()} wines)")
                st.write(f"**Least Common Quality:** {quality_counts.idxmin()} ({quality_counts.min()} wines)")
                st.write(f"**Quality Distribution:**")
                for quality, count in quality_counts.items():
                    percentage = (count / len(df)) * 100
                    st.write(f"Quality {quality}: {count} wines ({percentage:.1f}%)")
        
        with tab2:
            fig, ax = plt.subplots(figsize=(14, 10))
            correlation_matrix = df.corr()
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', 
                       cmap='RdYlBu_r', center=0, square=True, ax=ax,
                       cbar_kws={"shrink": .8})
            ax.set_title('Wine Feature Correlation Matrix', fontsize=16, pad=20)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Show top correlations with quality
            st.markdown("### 🔗 Top Correlations with Quality")
            quality_corr = correlation_matrix['quality'].drop('quality').sort_values(ascending=False)
            for feature, corr in quality_corr.items():
                st.write(f"**{feature}:** {corr:.3f}")
        
        with tab3:
            st.markdown("### 📊 Statistical Summary")
            st.dataframe(df.describe().round(3))
            
            st.markdown("### 📈 Feature Ranges")
            for col in df.columns:
                min_val = df[col].min()
                max_val = df[col].max()
                mean_val = df[col].mean()
                st.write(f"**{col}:** {min_val:.2f} - {max_val:.2f} (avg: {mean_val:.2f})")
        
        with tab4:
            st.markdown("### 🔍 Sample Wine Data")
            st.dataframe(df.head(15))
            
            st.download_button(
                label="📥 Download Sample Data",
                data=df.head(100).to_csv(index=False),
                file_name="wine_sample.csv",
                mime="text/csv"
            )
            
    except Exception as e:
        st.error(f"Error loading data: {e}")

elif page == "🤖 Model Info":
    st.header("🤖 Model Information & Performance")
    
    st.markdown("""
    ### 🎯 About the Model
    This wine quality prediction system uses advanced machine learning algorithms to predict 
    wine quality based on physicochemical properties. The model was trained on 1,599 red wines 
    with quality ratings from 3 to 8.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Model Features")
        features = [
            "**Fixed Acidity** - Tartaric acid content",
            "**Volatile Acidity** - Acetic acid content",
            "**Citric Acid** - Citric acid content",
            "**Residual Sugar** - Remaining sugar",
            "**Chlorides** - Sodium chloride content",
            "**Free SO₂** - Free sulfur dioxide",
            "**Total SO₂** - Total sulfur dioxide",
            "**Density** - Wine density",
            "**pH Level** - Acidity level",
            "**Sulphates** - Potassium sulfate",
            "**Alcohol** - Alcohol content"
        ]
        
        for feature in features:
            st.write(f"• {feature}")
    
    with col2:
        st.markdown("### 🎯 How to Use")
        st.write("""
        1. **Navigate** to the Predict Quality page
        2. **Adjust** sliders to match wine characteristics
        3. **Click** Predict Quality for instant results
        4. **Analyze** feature comparisons and insights
        5. **Download** data and results as needed
        """)
        
        st.markdown("### 📱 Tips")
        st.info("""
        - Use the Data Analysis tab for dataset insights
        - Compare your wine against averages
        - Check model performance metrics
        - Download sample data for further analysis
        """)

elif page == "📖 Usage Guide":
    st.header("📖 Complete Usage Guide")
    
    st.markdown("""
    ### 🚀 Getting Started
    
    This wine quality predictor is designed to be intuitive and user-friendly. 
    Follow these steps to get the most out of the application.
    """)
    
    # Create expandable sections
    with st.expander("🎯 Making Predictions"):
        st.markdown("""
        1. **Navigate** to the Predict Quality page
        2. **Adjust sliders** to match your wine's characteristics
        3. **Click** the Predict Quality button
        4. **Review** the predicted quality score
        5. **Analyze** the feature comparison charts
        6. **Interpret** the quality assessment
        """)
    
    with st.expander("📊 Understanding Results"):
        st.markdown("""
        **Quality Scale:**
        - **8-10**: Exceptional quality
        - **7-8**: Very good quality
        - **6-7**: Good quality
        - **5-6**: Average quality
        - **3-5**: Below average quality
        
        **Feature Analysis:**
        - Compare your wine against dataset averages
        - Identify which features contribute most
        - Understand how each characteristic affects quality
        """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 🍷 About")
st.sidebar.info("""
**Wine Quality Predictor**
- Advanced ML-powered predictions
- Interactive visualizations
- Comprehensive data analysis
- User-friendly interface
""")

# Add footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; margin-top: 2rem;'>
        <p>🍷 <strong>Wine Quality Predictor</strong> | Built with Streamlit & Machine Learning</p>
        <p>Made with ❤️ for wine enthusiasts and data scientists</p>
    </div>
    """,
    unsafe_allow_html=True
)
