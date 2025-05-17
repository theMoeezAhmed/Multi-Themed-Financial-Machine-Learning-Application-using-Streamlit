import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import datetime
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_message
import uuid
import io

# Set page config ONCE at the very top
st.set_page_config(page_title="Market Master", layout="wide", page_icon="💹")

# Initialize session state
def init_session_state():
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = {
            'current_step': 0, 'data_loaded': False, 'preprocessed': False, 'features_engineered': False,
            'data_split': False, 'model_trained': False, 'model_evaluated': False, 'results_visualized': False,
            'df': None, 'df_processed': None, 'target': None, 'features': None,
            'X_train': None, 'X_test': None, 'y_train': None, 'y_test': None,
            'models': {}, 'y_preds': {}, 'current_price': None, 'last_symbol': None
        }
    if 'theme' not in st.session_state:
        st.session_state.theme = "Financial Shinobi"

# Define themes
def get_theme_css():
    # Make sure theme is initialized
    if 'theme' not in st.session_state:
        st.session_state.theme = "Financial Shinobi"
        
    themes = {
        "Financial Shinobi": """
    @import url('https://fonts.cdnfonts.com/css/anime-ace');
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');

    :root {
        --main-bg: #181c24;
        --card-bg: #23293a;
        --accent-crimson: #B22222;
        --accent-purple: #8A2BE2;
        --accent-neon: #39FF14;
        --text-main: #F8F8FF;
        --text-muted: #b0b0b0;
        --border-main: #8A2BE2;
        --border-accent: #39FF14;
    }

    .stApp {
        background: linear-gradient(135deg, var(--main-bg), #23293a 80%), url('https://www.transparenttextures.com/patterns/stardust.png');
        background-blend-mode: overlay;
        color: var(--text-main);
        font-family: 'Anime Ace', 'Comic Neue', cursive;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #FFF;
        background: linear-gradient(135deg, #181c24 60%, #23293a 100%);
        border: 2px solid var(--accent-purple);
        border-radius: 12px;
        padding: 12px 24px;
        font-family: 'Anime Ace', 'Comic Neue', cursive;
        font-weight: 700;
        margin-bottom: 15px;
        text-shadow: none;
        box-shadow: 0 2px 8px rgba(138,43,226,0.15);
        display: inline-block;
    }

    /* Card/Container backgrounds */
    .css-1d391kg, .stDataFrame, .stPlotlyChart, .stExpander, .stSelectbox, .stMultiSelect, .stFileUploader, .stNumberInput, .stSlider {
        background: var(--card-bg) !important;
        border-radius: 15px !important;
        border: 2px solid var(--border-main) !important;
        color: var(--text-main) !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.6);
    }

    /* Buttons */
    .stButton>button, .reset-button>button {
        background: linear-gradient(135deg, var(--accent-crimson), var(--accent-purple));
        color: var(--text-main);
        border: 2px solid var(--border-accent);
        padding: 12px 22px;
        border-radius: 10px;
        margin: 8px 0;
        width: 100%;
        font-size: 18px;
        font-family: 'Anime Ace', 'Comic Neue', cursive;
        transition: all 0.2s ease;
        box-shadow: 0 2px 6px rgba(0,0,0,0.4);
    }
    .stButton>button:hover, .reset-button>button:hover {
        background: var(--accent-neon);
        color: #181c24;
        border-color: var(--accent-purple);
        transform: translateY(-2px) scale(1.04);
        box-shadow: 0 0 10px var(--accent-purple);
    }
    .stButton>button:disabled {
        background: #444;
        border-color: #666;
        color: #aaa;
        cursor: not-allowed;
        opacity: 0.6;
    }

    /* Sidebar progress bar */
    .progress-bar {
        background: #23293a;
        border-radius: 8px;
        height: 15px;
        margin: 15px 0;
        border: 1.5px solid var(--accent-purple);
        overflow: hidden;
        position: relative;
    }
    .progress-fill {
        background: linear-gradient(90deg, var(--accent-crimson), var(--accent-purple));
        height: 100%;
        border-radius: 8px;
        transition: width 0.5s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 0 16px 4px #39FF14, 0 0 32px 8px #8A2BE2;
        animation: progressGlow 2s infinite alternate;
    }
    @keyframes progressGlow {
        from { box-shadow: 0 0 8px #39FF14, 0 0 16px #8A2BE2; }
        to   { box-shadow: 0 0 24px #39FF14, 0 0 32px #8A2BE2; }
    }

    /* Download button */
    .stDownloadButton>button {
        background: var(--accent-neon);
        color: #181c24;
        border: 2px solid var(--accent-crimson);
        padding: 10px 20px;
        border-radius: 10px;
        transition: all 0.2s;
    }
    .stDownloadButton>button:hover {
        background: var(--accent-purple);
        color: var(--text-main);
        box-shadow: 0 0 10px var(--accent-neon);
    }

    /* Info/Success/Warning/Error boxes */
    .stInfo, .stSuccess, .stWarning, .stError {
        border-radius: 12px;
        border: 2px solid;
        padding: 15px;
        color: var(--text-main);
        font-family: 'Roboto', sans-serif;
    }
    .stInfo, .stSuccess {
        background: var(--card-bg);
        border-color: var(--accent-neon);
    }
    .stWarning {
        background: var(--card-bg);
        border-color: var(--accent-crimson);
    }
    .stError {
        background: var(--card-bg);
        border-color: var(--accent-purple);
    }

    /* Spinner */
    .stSpinner > div > div {
        border: 4px solid var(--accent-crimson);
        border-top: 4px solid var(--accent-neon);
        border-radius: 50%;
        width: 32px;
        height: 32px;
        animation: spin 0.8s linear infinite;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    /* Expander, select, input, slider, etc. */
    .stExpander, .stSelectbox, .stMultiSelect, .stFileUploader, .stNumberInput, .stSlider {
        background: var(--card-bg) !important;
        border: 2px solid var(--accent-crimson) !important;
        color: var(--text-main) !important;
    }

    /* DataFrame and Plotly chart tweaks */
    .stDataFrame, .stPlotlyChart {
        background: var(--card-bg) !important;
        border: 2px solid var(--accent-purple) !important;
        color: var(--text-main) !important;
    }

    /* Remove excessive animation from cards */
    .css-1d391kg, .stDataFrame, .stPlotlyChart {
        animation: none !important;
    }

    /* Welcome hero section */
    .welcome-hero {
        background: linear-gradient(rgba(24,28,36,0.95), rgba(35,41,58,0.95)), 
                            url('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQjp3muwfBgyjebUOVbOerdEEm8SP9yRaAgTg&s');
        background-size: cover;
        background-position: center;
        border-radius: 15px;
        padding: 40px 20px;
        text-align: center;
        border: 2px solid var(--accent-purple);
        box-shadow: 0 0 16px var(--accent-neon);
    }

    /* Centered images */
    .center-image {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 25px 0;
    }

    /* Interpretation cards */
    .interpretation {
        background: var(--card-bg);
        border-radius: 10px;
        padding: 12px 18px;
        margin-top: 12px;
        color: var(--text-main);
        border: 2px solid var(--accent-neon);
        font-family: 'Roboto', sans-serif;
    }

    /* Miscellaneous tweaks */
    .stMarkdown, .stText, .stSubheader, .stHeader, .stCaption, .stTable, .stDataFrame, .stPlotlyChart {
        color: var(--text-main) !important;
        font-family: 'Roboto', 'Anime Ace', 'Comic Neue', cursive, sans-serif !important;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(135deg, #181c24 60%, #23293a 100%);
        border-right: 3px solid var(--accent-purple);
        min-width: 320px;
        max-width: 400px;
        color: #FFFFFF;
        font-family: 'Anime Ace', 'Comic Neue', cursive;
        box-shadow: 4px 0 16px 0 rgba(57,255,20,0.08);
    }
    /* Sidebar header */
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3, 
    section[data-testid="stSidebar"] h4, 
    section[data-testid="stSidebar"] h5, 
    section[data-testid="stSidebar"] h6 {
        color: #FFFFFF;
        font-family: 'Anime Ace', 'Comic Neue', cursive;
        font-weight: 700;
        text-shadow: none;
        margin-bottom: 10px;
        margin-top: 10px;
    }
    /* Sidebar divider */
    section[data-testid="stSidebar"] hr, .stSidebar .stDivider {
        border: none;
        border-top: 2px dashed var(--accent-neon);
        margin: 18px 0;
    }
    /* Sidebar buttons */
    section[data-testid="stSidebar"] .stButton>button {
        background: linear-gradient(90deg, var(--accent-purple), var(--accent-crimson)) !important;
        color: #FFF !important;
        border: 2px solid var(--accent-neon);
        border-radius: 10px;
        font-family: 'Anime Ace', 'Comic Neue', cursive;
        font-size: 17px;
        margin: 6px 0;
        box-shadow: 0 2px 8px rgba(138,43,226,0.15);
        transition: all 0.2s;
        font-weight: 700;
        opacity: 1 !important;
    }
    section[data-testid="stSidebar"] .stButton>button:disabled,
    section[data-testid="stSidebar"] .stButton button[disabled] {
        background: #23293a !important;
        background-color: #23293a !important;
        border-color: #444 !important;
        color: #888 !important;
        opacity: 1 !important;
        font-weight: 700;
    }
    /* Sidebar progress bar */
    section[data-testid="stSidebar"] .progress-bar {
        background: #23293a;
        border-radius: 8px;
        height: 15px;
        margin: 15px 0;
        border: 1.5px solid var(--accent-purple);
    }
    section[data-testid="stSidebar"] .progress-fill {
        background: linear-gradient(90deg, var(--accent-crimson), var(--accent-purple));
        height: 100%;
        border-radius: 8px;
        transition: width 0.5s ease-in-out;
        box-shadow: 0 0 8px var(--accent-neon);
    }
    /* Sidebar text - force all text to be white and remove shadows for clarity */
    section[data-testid="stSidebar"] * {
        color: #FFF !important;
        text-shadow: none !important;
    }
    /* Sidebar button highlight (active/focused/selected) */
    section[data-testid="stSidebar"] .stButton>button:focus,
    section[data-testid="stSidebar"] .stButton>button:active,
    section[data-testid="stSidebar"] .stButton>button:not(:disabled):hover {
        background: linear-gradient(90deg, var(--accent-purple), var(--accent-crimson));
        color: #FFF !important;
        border-color: var(--accent-neon);
        outline: none !important;
        box-shadow: 0 0 10px var(--accent-neon);
    }
    section[data-testid="stSidebar"] .stButton button {
        background: linear-gradient(90deg, #2E3B55, #181c24) !important;
        background-color: #23293a !important;
        color: #FFF !important;
        border: 2px solid #8A2BE2 !important;
        font-weight: 700 !important;
        box-shadow: none !important;
    }

    /* Streamlit main header (top bar) */
    header[data-testid="stHeader"] {
        background: linear-gradient(135deg, #181c24 60%, #23293a 100%) !important;
        color: #FFF !important;
        box-shadow: none !important;
        border-bottom: 2px solid #8A2BE2 !important;
    }
    header[data-testid="stHeader"] .stDeployButton {
        color: #FFF !important;
    }

    /* 1. Neon Glow Animation */
    .neon-glow {
        box-shadow: 0 0 8px #39FF14, 0 0 16px #8A2BE2;
        animation: neonPulse 1.5s infinite alternate;
    }
    @keyframes neonPulse {
        from { box-shadow: 0 0 8px #39FF14, 0 0 16px #8A2BE2; }
        to   { box-shadow: 0 0 24px #39FF14, 0 0 32px #8A2BE2; }
    }

    /* 2. Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        background: #181c24;
    }
    ::-webkit-scrollbar-thumb {
        background: #8A2BE2;
        border-radius: 8px;
        border: 2px solid #39FF14;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #B22222;
    }
    /* For Firefox */
    html {
        scrollbar-color: #8A2BE2 #181c24;
        scrollbar-width: thin;
    }

    /* 3. Fade-in Animation for Sections */
    .fade-in-section {
        opacity: 0;
        transform: translateY(20px);
        animation: fadeInSection 1s ease-out forwards;
    }
    @keyframes fadeInSection {
        to {
            opacity: 1;
            transform: none;
        }
    }

    /* 4. Anime-style Emojis/Icons: (handled in headers/buttons in code) */

    /* 5. Subtle Background Pattern/Texture */
    .stApp {
        background: linear-gradient(135deg, var(--main-bg), #23293a 80%), url('https://www.transparenttextures.com/patterns/stardust.png');
        background-blend-mode: overlay;
        color: var(--text-main);
        font-family: 'Anime Ace', 'Comic Neue', cursive;
    }

    /* 6. Custom Font for Numbers */
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');
    .anime-numbers, .stMetricValue, .stMetricLabel {
        font-family: 'Share Tech Mono', 'Roboto Mono', monospace !important;
        letter-spacing: 1px;
    }

    .neon-graph-container {
        border: 3px solid #8A2BE2;
        border-radius: 18px;
        box-shadow: 0 0 24px #39FF14, 0 0 48px #8A2BE2;
        padding: 18px;
        margin-bottom: 32px;
        background: rgba(24,28,36,0.95);
    }
        """,
        
        "Techno Exchange": """
            @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700&display=swap');
            :root {
                --main-bg: #0f2027;
                --card-bg: rgba(20,30,50,0.92);
                --accent-primary: #00fff7;
                --accent-secondary: #ff9800;
                --accent-success: #4caf50;
                --accent-danger: #f44336;
                --text-main: #e0f7fa;
                --text-muted: #888;
                --border-main: #00fff7;
                --shadow: 0 4px 24px rgba(0,255,247,0.10), 0 1.5px 8px rgba(0,0,0,0.04);
            }
            .stApp {
                background: linear-gradient(120deg, #0f2027 0%, #2c5364 80%);
                background-size: 400% 400%;
                animation: gradientBG 12s ease-in-out infinite;
                color: var(--text-main);
                font-family: 'Orbitron', 'Montserrat', 'Roboto', Arial, sans-serif;
            }
            @keyframes gradientBG {
                0% {background-position: 0% 50%;}
                50% {background-position: 100% 50%;}
                100% {background-position: 0% 50%;}
            }
            h1, h2, h3, h4, h5, h6 {
                color: var(--accent-primary) !important;
                background: rgba(20,30,50,0.92);
                border: none;
                border-radius: 16px;
                padding: 16px 32px;
                font-family: 'Orbitron', 'Montserrat', 'Roboto', Arial, sans-serif;
                font-weight: 700;
                margin-bottom: 16px;
                box-shadow: var(--shadow);
                display: inline-block;
                animation: fadeInCard 0.7s;
                letter-spacing: 1px;
                text-shadow: 0 0 8px #00fff7, 0 0 16px #ff9800;
            }
            @keyframes fadeInCard {
                from { opacity: 0; transform: translateY(20px); }
                to { opacity: 1; transform: none; }
            }
            section[data-testid=\"stSidebar\"] {
                background: rgba(20,30,50,0.92);
                border-right: 2px solid var(--accent-primary);
                min-width: 300px;
                max-width: 380px;
                color: #000 !important;
                font-family: 'Orbitron', 'Montserrat', 'Roboto', Arial, sans-serif;
                box-shadow: 2px 0 16px 0 rgba(0,255,247,0.08);
                backdrop-filter: blur(4px) saturate(120%);
            }
            section[data-testid=\"stSidebar\"] * {
                color: #000 !important;
                text-shadow: none !important;
            }
            section[data-testid=\"stSidebar\"] h1, 
            section[data-testid=\"stSidebar\"] h2, 
            section[data-testid=\"stSidebar\"] h3, 
            section[data-testid=\"stSidebar\"] h4, 
            section[data-testid=\"stSidebar\"] h5, 
            section[data-testid=\"stSidebar\"] h6 {
                color: #000 !important;
                font-family: 'Orbitron', 'Montserrat', 'Roboto', Arial, sans-serif;
                font-weight: 700;
                text-shadow: none !important;
                margin-bottom: 10px;
                margin-top: 10px;
            }
            header[data-testid=\"stHeader\"] {
                background: linear-gradient(120deg, #0f2027 0%, #2c5364 80%) !important;
                color: var(--accent-primary) !important;
                box-shadow: none !important;
                border-bottom: 2px solid var(--accent-primary) !important;
            }
            header[data-testid=\"stHeader\"] * {
                color: var(--accent-primary) !important;
                font-family: 'Orbitron', 'Montserrat', 'Roboto', Arial, sans-serif !important;
                text-shadow: 0 0 8px #00fff7, 0 0 16px #ff9800 !important;
            }
            .progress-bar {
                background: #1a2636;
                border-radius: 8px;
                height: 14px;
                margin: 14px 0;
                border: 1.5px solid var(--accent-primary);
                overflow: hidden;
                position: relative;
            }
            .progress-fill {
                background: linear-gradient(90deg, var(--accent-primary), var(--accent-secondary));
                height: 100%;
                border-radius: 8px;
                transition: width 0.3s ease-in-out;
                box-shadow: 0 0 8px var(--accent-primary);
            }
            .stButton>button, .reset-button>button {
                background: linear-gradient(90deg, var(--accent-primary), var(--accent-secondary));
                color: #fff;
                border: none;
                padding: 14px 28px;
                border-radius: 12px;
                margin: 10px 0;
                width: 100%;
                font-size: 18px;
                font-family: 'Orbitron', 'Montserrat', 'Roboto', Arial, sans-serif;
                transition: background 0.2s, box-shadow 0.2s, transform 0.2s;
                box-shadow: 0 2px 12px rgba(0,255,247,0.10);
                font-weight: 700;
                letter-spacing: 0.5px;
            }
            .stButton>button:hover, .reset-button>button:hover {
                background: linear-gradient(90deg, var(--accent-secondary), var(--accent-primary));
                color: #fff;
                box-shadow: 0 4px 24px rgba(255,152,0,0.15);
                transform: translateY(-2px) scale(1.03);
            }
            .stButton>button:disabled {
                background: #222b3a;
                color: #aaa;
                cursor: not-allowed;
                opacity: 0.7;
            }
            .stDownloadButton>button {
                background: var(--accent-success);
                color: #fff;
                border: none;
                padding: 12px 24px;
                border-radius: 10px;
                transition: background 0.2s;
                font-family: 'Orbitron', 'Montserrat', 'Roboto', Arial, sans-serif;
                font-weight: 700;
            }
            .stDownloadButton>button:hover {
                background: var(--accent-secondary);
                color: #fff;
                box-shadow: 0 0 10px var(--accent-primary);
            }
            .stInfo, .stSuccess, .stWarning, .stError {
                border-radius: 14px;
                border: 2px solid;
                padding: 18px;
                color: var(--text-main);
                font-family: 'Orbitron', 'Montserrat', 'Roboto', Arial, sans-serif;
                background: var(--card-bg);
            }
            .stInfo, .stSuccess {
                border-color: var(--accent-success);
            }
            .stWarning {
                border-color: var(--accent-danger);
            }
            .stError {
                border-color: var(--accent-secondary);
            }
            .stSpinner > div > div {
                border: 4px solid var(--accent-primary);
                border-top: 4px solid var(--accent-secondary);
                border-radius: 50%;
                width: 32px;
                height: 32px;
                animation: spin 0.8s linear infinite;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            .stExpander, .stSelectbox, .stMultiSelect, .stFileUploader, .stNumberInput, .stSlider {
                background: var(--card-bg) !important;
                border: 1.5px solid var(--accent-primary) !important;
                color: var(--text-main) !important;
                backdrop-filter: blur(4px) saturate(120%);
            }
            .stDataFrame, .stPlotlyChart {
                background: var(--card-bg) !important;
                border: 1.5px solid var(--accent-primary) !important;
                color: var(--text-main) !important;
                box-shadow: var(--shadow);
            }
            .css-1d391kg, .stDataFrame, .stPlotlyChart {
                animation: none !important;
            }
            .welcome-hero {
                background: linear-gradient(rgba(20,30,50,0.95), rgba(44,83,100,0.95)), url('https://media.giphy.com/media/26ufdipQqU2lhNA4g/giphy.gif');
                background-size: cover;
                background-position: center;
                border-radius: 18px;
                padding: 40px 20px;
                text-align: center;
                border: 2px solid var(--accent-primary);
                box-shadow: 0 0 16px var(--accent-secondary);
            }
            .center-image {
                display: flex;
                justify-content: center;
                align-items: center;
                margin: 25px 0;
            }
            .interpretation {
                background: var(--card-bg);
                border-radius: 12px;
                padding: 14px 22px;
                margin-top: 14px;
                color: var(--text-main);
                border: 2px solid var(--accent-primary);
                font-family: 'Orbitron', 'Montserrat', 'Roboto', Arial, sans-serif;
            }
            .stMarkdown, .stText, .stSubheader, .stHeader, .stCaption, .stTable, .stDataFrame, .stPlotlyChart {
                color: var(--text-main) !important;
                font-family: 'Orbitron', 'Montserrat', 'Roboto', Arial, sans-serif !important;
            }
            /* Custom Scrollbar */
            ::-webkit-scrollbar {
                width: 10px;
                background: #1a2636;
            }
            ::-webkit-scrollbar-thumb {
                background: var(--accent-primary);
                border-radius: 8px;
                border: 2px solid var(--accent-secondary);
            }
            ::-webkit-scrollbar-thumb:hover {
                background: var(--accent-secondary);
            }
            html {
                scrollbar-color: var(--accent-primary) #1a2636;
                scrollbar-width: thin;
            }
        """,
        "Imperial Wealth Club": """
            @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Cinzel:wght@700&display=swap');
            :root {
                --main-bg: #f5f3e7;
                --card-bg: #f9f6ef;
                --accent-gold: #bfa14a;
                --accent-green: #3a5a40;
                --accent-brown: #7c5e3c;
                --text-main: #2d2a22;
                --text-muted: #7c5e3c;
                --border-main: #bfa14a;
                --shadow: 0 2px 12px rgba(122, 98, 60, 0.08);
            }
            .stApp {
                background: var(--main-bg);
                color: var(--text-main);
                font-family: 'Playfair Display', 'Cinzel', Georgia, serif;
            }
            h1, h2, h3, h4, h5, h6 {
                color: var(--accent-gold);
                background: var(--card-bg);
                border: 2px solid var(--accent-gold);
                border-radius: 8px;
                padding: 14px 28px;
                font-family: 'Cinzel', 'Playfair Display', Georgia, serif;
                font-weight: 700;
                margin-bottom: 18px;
                text-shadow: 0 1px 0 #fff8dc;
                box-shadow: var(--shadow);
                display: inline-block;
                animation: fadeInTicker 1.2s;
            }
            @keyframes fadeInTicker {
                0% { opacity: 0; transform: translateX(-60px); }
                60% { opacity: 0.7; transform: translateX(10px); }
                100% { opacity: 1; transform: none; }
            }
            section[data-testid=\"stSidebar\"] {
                background: var(--card-bg);
                border-right: 2px solid var(--accent-gold);
                min-width: 320px;
                max-width: 400px;
                color: var(--text-main);
                font-family: 'Cinzel', 'Playfair Display', Georgia, serif;
                box-shadow: 4px 0 24px 0 #bfa14a22;
            }
            section[data-testid=\"stSidebar\"] * {
                color: var(--accent-green) !important;
                text-shadow: 0 1px 0 #fff8dc !important;
            }
            section[data-testid=\"stSidebar\"] h1, 
            section[data-testid=\"stSidebar\"] h2, 
            section[data-testid=\"stSidebar\"] h3, 
            section[data-testid=\"stSidebar\"] h4, 
            section[data-testid=\"stSidebar\"] h5, 
            section[data-testid=\"stSidebar\"] h6 {
                color: var(--accent-gold) !important;
                font-family: 'Cinzel', 'Playfair Display', Georgia, serif;
                font-weight: 700;
                text-shadow: 0 1px 0 #fff8dc !important;
                margin-bottom: 10px;
                margin-top: 10px;
            }
            header[data-testid=\"stHeader\"] {
                background: var(--main-bg) !important;
                color: var(--accent-gold) !important;
                box-shadow: none !important;
                border-bottom: 2px solid var(--accent-gold) !important;
            }
            header[data-testid=\"stHeader\"] * {
                color: var(--accent-gold) !important;
                font-family: 'Cinzel', 'Playfair Display', Georgia, serif !important;
                text-shadow: 0 1px 0 #fff8dc !important;
            }
            .progress-bar {
                background: #e7e2c7;
                border-radius: 8px;
                height: 14px;
                margin: 14px 0;
                border: 1.5px solid var(--accent-gold);
                overflow: hidden;
                position: relative;
            }
            .progress-fill {
                background: linear-gradient(90deg, var(--accent-gold), var(--accent-green));
                height: 100%;
                border-radius: 8px;
                transition: width 0.5s cubic-bezier(0.4, 0, 0.2, 1);
            }
            /* ... rest of the theme CSS ... */
        """
    }
    
    return themes[st.session_state.theme]

# Theme-specific GIFs
THEME_GIFS = {
    "Financial Shinobi": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQjp3muwfBgyjebUOVbOerdEEm8SP9yRaAgTg&s",
    "Techno Exchange": "https://media.giphy.com/media/26ufdipQqU2lhNA4g/giphy.gif",
    "Imperial Wealth Club": "https://media.giphy.com/media/3o6Zt6ML6BklcajjsA/giphy.gif"
}

# Theme-specific display names
THEME_TITLES = {
    "Financial Shinobi": " FINANCIAL SHINOBI ",
    "Techno Exchange": " TECHNO EXCHANGE PROTOCOL ",
    "Imperial Wealth Club": " IMPERIAL WEALTH CLUB "
}

# Theme-specific sidebar section titles
THEME_SIDEBAR_TITLES = {
    "Financial Shinobi": "⚔️ Training Stages",
    "Techno Exchange": "🟦 Protocol Steps",
    "Imperial Wealth Club": "👑 Club Ledger"
}

# Theme-specific main page headers
THEME_MAIN_HEADERS = {
    "Financial Shinobi": " FINANCIAL SHINOBI ",
    "Techno Exchange": " TECHNO EXCHANGE PROTOCOL ",
    "Imperial Wealth Club": " IMPERIAL WEALTH CLUB "
}

# Expanded theme-specific step headers and button labels for all steps
THEME_STEP_LABELS = {
    "welcome": {
        "Financial Shinobi": ("Welcome, Shinobi!", "⚔️ Begin Your Journey"),
        "Techno Exchange": ("System Online. Welcome, Operator!", "🟦 Initialize Protocol"),
        "Imperial Wealth Club": ("Welcome, Esteemed Financier!", "👑 Enter the Treasury")
    },
    "load_data": {
        "Financial Shinobi": ("📜 Step 1: Summon Cursed Scrolls", "⚔️ Scroll Summoned!", "⚔️ Scrolls Summoned!"),
        "Techno Exchange": ("🟦 Step 1: Upload Data Stream", "🟦 Stream Uploaded!", "🟦 Data Streamed!"),
        "Imperial Wealth Club": ("👑 Step 1: Record Entry", "👑 Entry Recorded!", "👑 Data Retrieved!")
    },
    "preprocess": {
        "Financial Shinobi": ("💀 Step 2: Purify Cursed Scrolls", "Next ➡️"),
        "Techno Exchange": ("🟦 Step 2: Clean Data Matrix", "Run Clean Cycle ➡️"),
        "Imperial Wealth Club": ("👑 Step 2: Audit Accounts", "Continue ➡️")
    },
    "feature_engineering": {
        "Financial Shinobi": ("🩸 Step 3: Forge Blood Jutsu", "Next ➡️"),
        "Techno Exchange": ("🟦 Step 3: Engineer Quantum Features", "Engineer Features ➡️"),
        "Imperial Wealth Club": ("👑 Step 3: Calculate Dividends", "Continue ➡️")
    },
    "split": {
        "Financial Shinobi": ("⚔️ Step 4: Divide Shadow Clans", "Next ➡️"),
        "Techno Exchange": ("🟦 Step 4: Split Data Grid", "Split Grid ➡️"),
        "Imperial Wealth Club": ("👑 Step 4: Partition Holdings", "Continue ➡️")
    },
    "train": {
        "Financial Shinobi": ("🥷 Step 5: Train Dark Sensei", "Master Dark Jutsu", "Next ➡️"),
        "Techno Exchange": ("🟦 Step 5: Train AI Node", "Train AI", "Proceed ➡️"),
        "Imperial Wealth Club": ("👑 Step 5: Train the Banker", "Train Banker", "Continue ➡️")
    },
    "evaluate": {
        "Financial Shinobi": ("⚡️ Step 6: Unleash Forbidden Prophecies", "Next ➡️"),
        "Techno Exchange": ("🟦 Step 6: Evaluate Output", "Evaluate ➡️"),
        "Imperial Wealth Club": ("👑 Step 6: Review the Ledger", "Continue ➡️")
    },
    "results": {
        "Financial Shinobi": ("📜 Step 7: Reveal Forbidden Scrolls",),
        "Techno Exchange": ("🟦 Step 7: Visualize Protocol Insights",),
        "Imperial Wealth Club": ("👑 Step 7: Reveal the Treasury",)
    }
}

# Expanded theme-specific graph titles and axis labels for all steps
THEME_GRAPH_LABELS = {
    "price_chart": {
        "Financial Shinobi": ("Blood Price Chronicle", "Date", "Close"),
        "Techno Exchange": ("Neon Price Stream", "Timestamp", "Signal Value"),
        "Imperial Wealth Club": ("Wealth Over Time", "Date", "Wealth")
    },
    "volume_chart": {
        "Financial Shinobi": ("Chaos Frenzy Scroll", "Date", "Volume"),
        "Techno Exchange": ("Signal Volume", "Timestamp", "Volume"),
        "Imperial Wealth Club": ("Volume of Trades", "Date", "Volume")
    },
    "correlation_matrix": {
        "Financial Shinobi": ("Jutsu Blood Matrix", "", ""),
        "Techno Exchange": ("Quantum Correlation Grid", "", ""),
        "Imperial Wealth Club": ("Correlation of Holdings", "", "")
    },
    "scatter_matrix": {
        "Financial Shinobi": ("Jutsu Clash Plot", "Jutsu", "Jutsu"),
        "Techno Exchange": ("Feature Signal Matrix", "Signal", "Signal"),
        "Imperial Wealth Club": ("Feature Matrix", "Feature", "Feature")
    },
    # Add more for all other graphs in each step
}

# Expanded theme-specific interpretation texts for all steps/graphs
THEME_INTERPRETATIONS = {
    "load_data_price": {
        "Financial Shinobi": "This chronicle traces the blood price's path. Rising lines mark victories, falling lines reveal defeats. Hover to uncover hidden truths!",
        "Techno Exchange": "This neon stream visualizes asset price evolution. Uptrends signal bullish protocols, downtrends show system corrections. Hover for data points!",
        "Imperial Wealth Club": "This chart chronicles the growth of your estate. Peaks mark prosperous eras, valleys warn of lean times. Hover for historical context!"
    },
    "load_data_volume": {
        "Financial Shinobi": "Spikes in frenzy signal chaotic battles, often tied to price shifts (check the chronicle). Quiet frenzies suggest a lull in the war.",
        "Techno Exchange": "Volume spikes indicate high data throughput, often preceding price moves. Low volume means a quiet protocol.",
        "Imperial Wealth Club": "Tall bars show bustling trading days in the club. Calm periods reflect market stability."
    },
    "correlation_matrix": {
        "Financial Shinobi": "This crimson matrix reveals jutsu bonds. Values near 1 or -1 show strong blood pacts, near 0 suggests weak ties. Strong jutsu bonds may challenge your sensei.",
        "Techno Exchange": "This quantum grid shows feature correlations. High values indicate strong digital relationships. Use this to optimize your AI node.",
        "Imperial Wealth Club": "This matrix reveals how your holdings relate. Strong correlations may indicate redundant assets. Diversify for a robust portfolio."
    },
    "scatter_matrix": {
        "Financial Shinobi": "This clash plot shows jutsu battles. Diagonal scrolls are histograms. Linear sparks hint at blood pacts (check the matrix). Outliers or clans may shape your strategy.",
        "Techno Exchange": "Signal matrix visualizes feature relationships. Diagonals are distributions. Patterns may reveal clusters or outliers in the data stream.",
        "Imperial Wealth Club": "This matrix displays the spread of your club's features. Patterns may reveal trends or anomalies in your accounts."
    },
    # Add more for all other steps/graphs
}

# Theme-specific sidebar step names
THEME_SIDEBAR_STEPS = {
    "Financial Shinobi": [
        "📋 Begin Journey", "📜 Gather Data", "💀 Cleanse Data", "🔥 Forge Features",
        "⚔️ Split Forces", "👺 Train Model", "⚡️ Test Powers", "🏮 Reveal Results"
    ],
    "Techno Exchange": [
        "🟦 Boot Protocol", "🟦 Upload Data Stream", "🧹 Clean Data Matrix", "🧬 Engineer Quantum Features",
        "🔀 Split Data Grid", "🤖 Train AI Node", "📊 Evaluate Output", "🛰️ Visualize Insights"
    ],
    "Imperial Wealth Club": [
        "📖 Open the Ledger", "👑 Record Entry", "🧾 Audit Accounts", "💹 Calculate Dividends",
        "📂 Partition Holdings", "🏦 Train the Banker", "🔍 Review the Ledger", "🏛️ Reveal the Treasury"
    ]
}

# Theme-specific checkbox labels
THEME_CHECKBOX_LABELS = {
    "Financial Shinobi": "Apply Jutsu Scaling",
    "Techno Exchange": "Normalize Data Streams",
    "Imperial Wealth Club": "Standardize Holdings"
}

# Theme-specific expander titles
THEME_EXPANDER_TITLES = {
    "Financial Shinobi": "View Cursed Scroll & Secrets",
    "Techno Exchange": "View Data Stream & Node Stats",
    "Imperial Wealth Club": "View Ledger Page & Details"
}

# Helper functions
def clean_numeric_columns(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce')
            except Exception as e:
                st.warning(f"Could not convert {col} to numeric: {e}")
    return df

def is_continuous(series):
    return pd.api.types.is_numeric_dtype(series) and len(series.unique()) > 10

@st.cache_data
def fetch_yfinance_data(symbol, start_date, end_date, _cache_key=None):
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10), 
           retry=retry_if_exception_message(match='Too Many Requests'))
    def fetch():
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(start=start_date, end=end_date)
            if df.empty:
                st.error(f"No data for {symbol}. Try AAPL, TSLA, MSFT.")
                return None
            return df.reset_index()[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return None
    return fetch()

def fetch_current_price(symbol):
    try:
        stock = yf.Ticker(symbol)
        price = stock.info.get('regularMarketPrice', stock.info.get('currentPrice'))
        return price
    except Exception as e:
        st.warning(f"Could not fetch price for {symbol}: {e}")
        return None

def plot_config(fig, title, x_title, y_title, width=800, height=400):
    theme = st.session_state.theme
    if theme == "Techno Exchange":
        fig.update_layout(
            title=dict(text=title, font=dict(size=22, family="'Montserrat', 'Roboto', 'Inter', Arial, sans-serif", color="#00bcd4")),
            xaxis_title=dict(text=x_title, font=dict(size=16, family="'Montserrat', 'Roboto', 'Inter', Arial, sans-serif", color="#222")),
            yaxis_title=dict(text=y_title, font=dict(size=16, family="'Montserrat', 'Roboto', 'Inter', Arial, sans-serif", color="#222")),
            width=width, height=height, template='plotly_white',
            font_color='#222', plot_bgcolor='#f9f9fb', paper_bgcolor='#e0f7fa',
            showlegend=True, legend=dict(font=dict(size=14, family="'Montserrat', 'Roboto', 'Inter', Arial, sans-serif", color="#00bcd4"), x=0.01, y=0.99),
            hovermode='closest', margin=dict(l=50, r=50, t=50, b=50)
        )
        fig.update_xaxes(showgrid=True, gridcolor='rgba(0,188,212,0.15)', zeroline=False, color='#00bcd4')
        fig.update_yaxes(showgrid=True, gridcolor='rgba(0,188,212,0.15)', color='#00bcd4')
        for trace in fig.data:
            if trace.type in ['scatter', 'bar', 'scattergl', 'scatter3d', 'scatterpolar', 'scattergeo', 'scattermapbox']:
                trace.update(marker=dict(line=dict(color='#ff9800', width=2)))
    elif theme == "Imperial Wealth Club":
        fig.update_layout(
            title=dict(text=title, font=dict(size=22, family="'Playfair Display', 'Cinzel', serif", color="#bfa14a")),
            xaxis_title=dict(text=x_title, font=dict(size=16, family="'Playfair Display', 'Cinzel', serif", color="#3a5a40")),
            yaxis_title=dict(text=y_title, font=dict(size=16, family="'Playfair Display', 'Cinzel', serif", color="#3a5a40")),
            width=width, height=height, template='plotly_white',
            font_color='#2d2a22', plot_bgcolor='#f9f6ef', paper_bgcolor='#f5f3e7',
            showlegend=True, legend=dict(font=dict(size=14, family="'Playfair Display', 'Cinzel', serif", color="#bfa14a"), x=0.01, y=0.99),
            hovermode='closest', margin=dict(l=50, r=50, t=50, b=50)
        )
        fig.update_xaxes(showgrid=True, gridcolor='rgba(191,161,74,0.15)', zeroline=False, color='#bfa14a')
        fig.update_yaxes(showgrid=True, gridcolor='rgba(191,161,74,0.15)', color='#bfa14a')
        for trace in fig.data:
            if trace.type in ['scatter', 'bar', 'scattergl', 'scatter3d', 'scatterpolar', 'scattergeo', 'scattermapbox']:
                trace.update(marker=dict(line=dict(color='#3a5a40', width=2)))
    else:
        # Financial Shinobi (default)
        fig.update_layout(
            title=dict(text=title, font=dict(size=20, family="'Anime Ace', 'Comic Neue', cursive", color="#FFF")),
            xaxis_title=dict(text=x_title, font=dict(size=16, family="'Anime Ace', 'Comic Neue', cursive", color="#FFF")),
            yaxis_title=dict(text=y_title, font=dict(size=16, family="'Anime Ace', 'Comic Neue', cursive", color="#FFF")),
            width=width, height=height, template='plotly_dark',
            font_color='#FFF', plot_bgcolor='#181c24', paper_bgcolor='#23293a',
            showlegend=True, legend=dict(font=dict(size=14, family="'Anime Ace', 'Comic Neue', cursive", color="#39FF14"), x=0.01, y=0.99),
            hovermode='closest', margin=dict(l=50, r=50, t=50, b=50)
        )
        fig.update_xaxes(showgrid=True, gridcolor='rgba(138,43,226,0.3)', zeroline=False, color='#FFF')
        fig.update_yaxes(showgrid=True, gridcolor='rgba(138,43,226,0.3)', color='#FFF')
        for trace in fig.data:
            if trace.type in ['scatter', 'bar', 'scattergl', 'scatter3d', 'scatterpolar', 'scattergeo', 'scattermapbox']:
                trace.update(marker=dict(line=dict(color='#39FF14', width=2)))

# Pipeline steps
def welcome_step():
    theme = st.session_state.theme
    gif_url = THEME_GIFS.get(theme, THEME_GIFS["Financial Shinobi"])
    main_header = THEME_MAIN_HEADERS.get(theme, "Market Master")
    welcome_text, welcome_btn = THEME_STEP_LABELS["welcome"][theme]
    st.markdown(f'''
        <div style="display: flex; flex-direction: column; align-items: center; margin-bottom: 0;">
            <img src="{gif_url}" width="300" style="border: 4px solid #39FF14; border-radius: 18px; box-shadow: 0 0 24px #8A2BE2, 0 0 48px #39FF14; margin-bottom: 0;" />
            <h1 class="neon-glow" style="margin-top: 10px; text-align: center;">{main_header}</h1>
        </div>
    ''', unsafe_allow_html=True)
    st.markdown(f'''<div style="display: flex; align-items: center;" class="fade-in-section">
        <span class="neon-glow" style="font-size: 2rem;">{welcome_text}</span>
    </div>''', unsafe_allow_html=True)

    if st.button(welcome_btn, key="start"):
        st.session_state.pipeline['current_step'] = 1
        st.rerun()

def load_data_step():
    theme = st.session_state.theme
    header, upload_success, fetch_success = THEME_STEP_LABELS["load_data"][theme]
    st.header(header)
    data_option = st.radio("Scroll Source", ("Upload CSV/Excel", "Yahoo Finance"), help="Choose to upload a cursed scroll or summon live market data.")
    
    if data_option == "Upload CSV/Excel":
        uploaded_file = st.file_uploader("Upload Cursed Scroll 📜", type=["csv", "xlsx"], help="Upload a dataset with 'Date' and 'Close' columns.")
        if uploaded_file:
            try:
                df = clean_numeric_columns(pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file))
                if not {'Date', 'Close'}.issubset(df.columns):
                    st.warning("Scroll needs 'Date' and 'Close' seals.")
                st.session_state.pipeline.update({'df': df, 'data_loaded': True, 'last_symbol': None, 'current_step': 2})
                st.success(upload_success)
                with st.expander(THEME_EXPANDER_TITLES[theme]):
                    st.dataframe(df)
                    buffer = io.StringIO()
                    df.info(buf=buffer)
                    st.text(buffer.getvalue())
                    st.write("Secrets:")
                    st.dataframe(df.describe())
                if 'Date' in df.columns and 'Close' in df.columns:
                    price_title, price_x, price_y = THEME_GRAPH_LABELS['price_chart'][theme]
                    fig = px.line(df, x='Date', y='Close', title=price_title, color_discrete_sequence=['#B22222'], hover_data=['Close'])
                    plot_config(fig, price_title, price_x, price_y)
                    st.markdown(f"""
                        <div class="interpretation">
                        {THEME_INTERPRETATIONS['load_data_price'][theme]}
                        </div>
                    """, unsafe_allow_html=True)
                if 'Volume' in df.columns:
                    vol_title, vol_x, vol_y = THEME_GRAPH_LABELS['volume_chart'][theme]
                    fig = px.line(df, x='Date', y='Volume', title=vol_title, color_discrete_sequence=['#8A2BE2'], hover_data=['Volume'])
                    plot_config(fig, vol_title, vol_x, vol_y)
                    st.plotly_chart(fig)
                    st.markdown(f"""
                        <div class="interpretation">
                        {THEME_INTERPRETATIONS['load_data_volume'][theme]}
                        </div>
                    """, unsafe_allow_html=True)
                st.rerun()
            except Exception as e:
                st.error(f"❌ Summoning Failed: {e}")
    else:
        col1, col2 = st.columns(2)
        with col1:
            symbol = st.text_input("Stock Symbol (e.g., AAPL)", "AAPL", help="Enter a valid market seal.")
            start_date = st.date_input("Start Date", datetime.date(2024, 1, 1))
            end_date = st.date_input("End Date", datetime.date.today())
        with col2:
            if symbol and start_date < end_date:
                with st.spinner("Summoning Cursed Scrolls..."):
                    df = fetch_yfinance_data(symbol.upper(), start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), f"{symbol}_{start_date}_{end_date}")
                if df is not None:
                    price = fetch_current_price(symbol.upper())
                    if price:
                        st.metric(f"Current Blood Price ({symbol.upper()})", f"${price:.2f}")
                    st.session_state.pipeline.update({
                        'df': df, 'data_loaded': True, 'last_symbol': symbol.upper(), 'current_price': price, 'current_step': 2
                    })
                    st.success(fetch_success)
                    with st.expander(THEME_EXPANDER_TITLES[theme]):
                        st.dataframe(df)
                        buffer = io.StringIO()
                        df.info(buf=buffer)
                        st.text(buffer.getvalue())
                        st.write("Secrets:")
                        st.dataframe(df.describe())
                    price_title, price_x, price_y = THEME_GRAPH_LABELS['price_chart'][theme]
                    fig = go.Figure(data=[go.Candlestick(
                        x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
                        increasing_line_color='#B22222', decreasing_line_color='#8A2BE2'
                    )])
                    plot_config(fig, price_title, price_x, price_y)
                    st.plotly_chart(fig)
                    st.markdown(f"""
                        <div class="interpretation">
                        {THEME_INTERPRETATIONS['load_data_price'][theme]}
                        </div>
                    """, unsafe_allow_html=True)
                    vol_title, vol_x, vol_y = THEME_GRAPH_LABELS['volume_chart'][theme]
                    fig = px.line(df, x='Date', y='Volume', title=vol_title, color_discrete_sequence=['#8A2BE2'], hover_data=['Volume'])
                    plot_config(fig, vol_title, vol_x, vol_y)
                    st.plotly_chart(fig)
                    st.markdown(f"""
                        <div class="interpretation">
                        {THEME_INTERPRETATIONS['load_data_volume'][theme]}
                        </div>
                    """, unsafe_allow_html=True)
                    st.rerun()
            else:
                st.warning("Invalid seal or time scroll!")

def preprocessing_step():
    theme = st.session_state.theme
    header, next_btn = THEME_STEP_LABELS["preprocess"][theme]
    st.header(header)
    if not st.session_state.pipeline['data_loaded']:
        st.warning("Summon scrolls first!")
        return
    df = st.session_state.pipeline['df'].copy()
    
    missing_values = df.isnull().sum()
    if missing_values.sum():
        st.dataframe(missing_values[missing_values > 0].to_frame(name="Missing Values"))
        df[df.select_dtypes(np.number).columns] = df.select_dtypes(np.number).fillna(df.mean(numeric_only=True))
        st.success({
            "Financial Shinobi": "⚔️ Missing seals restored!",
            "Techno Exchange": "🧹 Missing values filled!",
            "Imperial Wealth Club": "🧾 Gaps reconciled!"
        }[theme])
    else:
        st.success({
            "Financial Shinobi": "⚔️ Scrolls are flawless!",
            "Techno Exchange": "💹 Data is clean!",
            "Imperial Wealth Club": "💰 Ledger is balanced!"
        }[theme])
    
    numeric_cols = df.select_dtypes(np.number).columns
    if numeric_cols.any():
        for col in numeric_cols:
            Q1, Q3 = df[col].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            df[col] = df[col].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
        st.success({
            "Financial Shinobi": "⚔️ Rogue seals banished!",
            "Techno Exchange": "🧹 Outliers handled!",
            "Imperial Wealth Club": "🧾 Outliers trimmed!"
        }[theme])
    
    st.session_state.pipeline.update({'df_processed': df, 'preprocessed': True})
    with st.expander({
        "Financial Shinobi": "View Purified Scrolls",
        "Techno Exchange": "View Cleaned Data",
        "Imperial Wealth Club": "View Audited Ledger"
    }[theme]):
        st.dataframe(df)
    if st.button(next_btn, key="preprocess_next"):
        st.session_state.pipeline['current_step'] = 3
        st.rerun()

def feature_engineering_step():
    theme = st.session_state.theme
    header, next_btn = THEME_STEP_LABELS["feature_engineering"][theme]
    st.header(header)
    if not st.session_state.pipeline['preprocessed']:
        st.warning({
            "Financial Shinobi": "Purify scrolls first!",
            "Techno Exchange": "Cleanse data first!",
            "Imperial Wealth Club": "Audit ledger first!"
        }[theme])
        return
    df = st.session_state.pipeline['df_processed'].copy()
    
    if 'Close' in df.columns:
        window = st.slider({
            "Financial Shinobi": "Jutsu Window (days)",
            "Techno Exchange": "Moving Average Window (days)",
            "Imperial Wealth Club": "Indicator Window (days)"
        }[theme], 5, 50, 20, help={
            "Financial Shinobi": "Select window for moving average and volatility.",
            "Techno Exchange": "Select window for moving average and volatility.",
            "Imperial Wealth Club": "Select window for rolling indicators."
        }[theme])
        df[f'MA_{window}'] = df['Close'].rolling(window=window).mean().fillna(df['Close'])
        df[f'Volatility_{window}'] = df['Close'].rolling(window=window).std().fillna(df['Close'].std())
        df['Daily_Return'] = df['Close'].pct_change().fillna(0)
        st.success({
            "Financial Shinobi": f"⚔️ Forged {window}-day MA, Volatility, Daily Return Jutsu!",
            "Techno Exchange": f"💹 Computed {window}-day MA, Volatility, Daily Return!",
            "Imperial Wealth Club": f"💰 Calculated {window}-day MA, Volatility, Daily Return!"
        }[theme])
    
    numeric_cols = df.select_dtypes(np.number).columns.tolist()
    if not numeric_cols:
        st.error({
            "Financial Shinobi": "No numeric jutsu found!",
            "Techno Exchange": "No numeric features found!",
            "Imperial Wealth Club": "No numeric indicators found!"
        }[theme])
        return
    
    col1, col2 = st.columns(2)
    with col1:
        target = st.selectbox({
            "Financial Shinobi": "Target Seal (y)",
            "Techno Exchange": "Target Variable (y)",
            "Imperial Wealth Club": "Target Entry (y)"
        }[theme], numeric_cols, index=numeric_cols.index('Close') if 'Close' in numeric_cols else 0, 
                              help={
            "Financial Shinobi": "Choose the seal for your prophecy.",
            "Techno Exchange": "Choose the variable to predict.",
            "Imperial Wealth Club": "Choose the ledger entry to forecast."
        }[theme])
    with col2:
        features = st.multiselect({
            "Financial Shinobi": "Jutsu (X)",
            "Techno Exchange": "Features (X)",
            "Imperial Wealth Club": "Indicators (X)"
        }[theme], [c for c in numeric_cols if c != target], 
                                  default=[c for c in numeric_cols if c != target][:2], 
                                  help={
            "Financial Shinobi": "Select jutsu for your battle.",
            "Techno Exchange": "Select features for your model.",
            "Imperial Wealth Club": "Select indicators for your analysis."
        }[theme])
    if not features:
        st.warning({
            "Financial Shinobi": "Select at least one jutsu!",
            "Techno Exchange": "Select at least one feature!",
            "Imperial Wealth Club": "Select at least one indicator!"
        }[theme])
        return
    
    if st.checkbox(THEME_CHECKBOX_LABELS[theme], value=True, help={
        "Financial Shinobi": "Sharpen jutsu for epic battles.",
        "Techno Exchange": "Normalize features for better model performance.",
        "Imperial Wealth Club": "Standardize indicators for fair comparison."
    }[theme]):
        try:
            df[features] = StandardScaler().fit_transform(df[features])
            st.success({
                "Financial Shinobi": "⚔️ Jutsu honed!",
                "Techno Exchange": "💹 Features normalized!",
                "Imperial Wealth Club": "💰 Indicators standardized!"
            }[theme])
        except Exception as e:
            st.error({
                "Financial Shinobi": f"❌ Jutsu error: {e}",
                "Techno Exchange": f"❌ Feature error: {e}",
                "Imperial Wealth Club": f"❌ Indicator error: {e}"
            }[theme])
    
    try:
        corr_title, corr_x, corr_y = THEME_GRAPH_LABELS['correlation_matrix'][theme]
        fig = px.imshow(df[features + [target]].corr(), text_auto=True, color_continuous_scale='Reds', title=corr_title, width=600, height=500)
        plot_config(fig, corr_title, corr_x, corr_y)
        st.plotly_chart(fig)
        st.markdown(f"""
            <div class="interpretation">
            {THEME_INTERPRETATIONS['correlation_matrix'][theme]}
            </div>
        """, unsafe_allow_html=True)
        scatter_title, scatter_x, scatter_y = THEME_GRAPH_LABELS['scatter_matrix'][theme]
        fig = px.scatter_matrix(df[features + [target]], title=scatter_title, width=800, height=600, color_discrete_sequence=['#39FF14'])
        plot_config(fig, scatter_title, scatter_x, scatter_y)
        st.plotly_chart(fig)
        st.markdown(f"""
            <div class="interpretation">
            {THEME_INTERPRETATIONS['scatter_matrix'][theme]}
            </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error({
            "Financial Shinobi": f"❌ Visualization jutsu failed: {e}",
            "Techno Exchange": f"❌ Visualization failed: {e}",
            "Imperial Wealth Club": f"❌ Visualization failed: {e}"
        }[theme])
    
    st.session_state.pipeline.update({'target': target, 'features': features, 'df_features': df, 'features_engineered': True})
    if st.button(next_btn, key="feature_next"):
        st.session_state.pipeline['current_step'] = 4
        st.rerun()

def train_test_split_step():
    theme = st.session_state.theme
    header, next_btn = THEME_STEP_LABELS["split"][theme]
    st.header(header)
    if not st.session_state.pipeline['features_engineered']:
        st.warning({
            "Financial Shinobi": "Forge jutsu first!",
            "Techno Exchange": "Engineer features first!",
            "Imperial Wealth Club": "Calculate indicators first!"
        }[theme])
        return
    df, target, features = (st.session_state.pipeline[k] for k in ['df_features', 'target', 'features'])
    
    col1, col2 = st.columns(2)
    with col1:
        test_size = st.slider({
            "Financial Shinobi": "Test Clan Size (%)",
            "Techno Exchange": "Test Set Size (%)",
            "Imperial Wealth Club": "Test Account Size (%)"
        }[theme], 10, 40, 20, help={
            "Financial Shinobi": "Percentage of shinobi for testing.",
            "Techno Exchange": "Percentage of data for testing.",
            "Imperial Wealth Club": "Percentage of accounts for testing."
        }[theme]) / 100
    with col2:
        random_state = st.number_input({
            "Financial Shinobi": "Shadow Seed",
            "Techno Exchange": "Random Seed",
            "Imperial Wealth Club": "Ledger Seed"
        }[theme], 0, 100, 42, help={
            "Financial Shinobi": "Seed for consistent clan splits.",
            "Techno Exchange": "Seed for reproducible splits.",
            "Imperial Wealth Club": "Seed for consistent account splits."
        }[theme])
    try:
        X, y = df[features].dropna(), df[target].loc[df[features].dropna().index]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        st.session_state.pipeline.update({
            'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test, 'data_split': True
        })
        pie_title = {
            "Financial Shinobi": "Training vs Testing Clans",
            "Techno Exchange": "Training vs Testing Sets",
            "Imperial Wealth Club": "Training vs Testing Accounts"
        }[theme]
        fig = px.pie(pd.DataFrame({'Set': [
            {"Financial Shinobi": "Training", "Techno Exchange": "Training", "Imperial Wealth Club": "Training"}[theme],
            {"Financial Shinobi": "Testing", "Techno Exchange": "Testing", "Imperial Wealth Club": "Testing"}[theme]
        ], 'Size': [len(X_train), len(X_test)]}), names='Set', values='Size', title=pie_title, width=400, height=400, color_discrete_sequence=['#B22222', '#8A2BE2'])
        plot_config(fig, pie_title, '', '')
        st.plotly_chart(fig)
        interp = {
            "Financial Shinobi": "This scroll divides your shinobi: training (crimson) for mastery, testing (purple) for trials. A larger training clan strengthens your jutsu, while the test clan ensures fair duels.",
            "Techno Exchange": "This pie shows the split between training and testing sets. More training data helps the model learn, while testing ensures fair evaluation.",
            "Imperial Wealth Club": "This chart divides your accounts: training (crimson) for model building, testing (purple) for validation. A larger training set improves learning."
        }[theme]
        st.markdown(f"""
            <div class="interpretation">
            {interp}
            </div>
        """, unsafe_allow_html=True)
        st.success({
            "Financial Shinobi": "⚔️ Clans divided!",
            "Techno Exchange": "💹 Data split complete!",
            "Imperial Wealth Club": "💰 Accounts partitioned!"
        }[theme])
        if st.button(next_btn, key="split_next"):
            st.session_state.pipeline['current_step'] = 5
            st.rerun()
    except Exception as e:
        st.error({
            "Financial Shinobi": f"❌ Clan division failed: {e}",
            "Techno Exchange": f"❌ Data split failed: {e}",
            "Imperial Wealth Club": f"❌ Account partition failed: {e}"
        }[theme])

def model_training_step():
    theme = st.session_state.theme
    header, train_btn, next_btn = THEME_STEP_LABELS["train"][theme]
    st.header(header)
    if not st.session_state.pipeline['data_split']:
        st.warning({
            "Financial Shinobi": "Divide clans first!",
            "Techno Exchange": "Split data first!",
            "Imperial Wealth Club": "Partition accounts first!"
        }[theme])
        return
    X_train, y_train = st.session_state.pipeline['X_train'], st.session_state.pipeline['y_train']
    model_options = [
        {"Financial Shinobi": "Linear Regression", "Techno Exchange": "Linear Regression", "Imperial Wealth Club": "Linear Regression"}[theme],
        {"Financial Shinobi": "Logistic Regression", "Techno Exchange": "Logistic Regression", "Imperial Wealth Club": "Logistic Regression"}[theme],
        {"Financial Shinobi": "K-Means Clustering", "Techno Exchange": "K-Means Clustering", "Imperial Wealth Club": "K-Means Clustering"}[theme]
    ]
    model_types = st.multiselect({
        "Financial Shinobi": "Select Sensei Jutsu",
        "Techno Exchange": "Select Model Type",
        "Imperial Wealth Club": "Select Analyst Method"
    }[theme], model_options, default=[model_options[0]], help={
        "Financial Shinobi": "Choose sensei jutsu to master.",
        "Techno Exchange": "Choose model type to train.",
        "Imperial Wealth Club": "Choose analyst method to train."
    }[theme])
    if not model_types:
        st.warning({
            "Financial Shinobi": "Select a sensei jutsu!",
            "Techno Exchange": "Select a model type!",
            "Imperial Wealth Club": "Select an analyst method!"
        }[theme])
        return
    target_is_continuous = is_continuous(y_train)
    if "Linear Regression" in model_types and not target_is_continuous:
        st.warning({
            "Financial Shinobi": "⚠️ Linear Regression needs continuous seals!",
            "Techno Exchange": "⚠️ Linear Regression needs continuous targets!",
            "Imperial Wealth Club": "⚠️ Linear Regression needs continuous entries!"
        }[theme])
        return
    if "Logistic Regression" in model_types and target_is_continuous:
        st.warning({
            "Financial Shinobi": "⚠️ Logistic Regression needs categorical seals!",
            "Techno Exchange": "⚠️ Logistic Regression needs categorical targets!",
            "Imperial Wealth Club": "⚠️ Logistic Regression needs categorical entries!"
        }[theme])
        return
    models = {}
    if "K-Means Clustering" in model_types:
        n_clusters = st.number_input({
            "Financial Shinobi": "Clans",
            "Techno Exchange": "Clusters",
            "Imperial Wealth Club": "Groups"
        }[theme], 2, 10, 3, key="clusters", help={
            "Financial Shinobi": "Number of clans for K-Means.",
            "Techno Exchange": "Number of clusters for K-Means.",
            "Imperial Wealth Club": "Number of groups for K-Means."
        }[theme])
        models["K-Means Clustering"] = KMeans(n_clusters=n_clusters, random_state=42)
    if "Linear Regression" in model_types:
        models["Linear Regression"] = LinearRegression()
    if "Logistic Regression" in model_types:
        models["Logistic Regression"] = LogisticRegression(max_iter=1000)
    if st.button(train_btn, key="train"):
        with st.spinner({
            "Financial Shinobi": "Training Sensei...",
            "Techno Exchange": "Training Model...",
            "Imperial Wealth Club": "Training Analyst..."
        }[theme]):
            try:
                for model_type, model in models.items():
                    model.fit(X_train, y_train if model_type != "K-Means Clustering" else X_train)
                st.session_state.pipeline.update({'models': models, 'model_trained': True})
                st.success({
                    "Financial Shinobi": "⚔️ Sensei mastered!",
                    "Techno Exchange": "💹 Model trained!",
                    "Imperial Wealth Club": "💰 Analyst trained!"
                }[theme])
                with st.expander({
                    "Financial Shinobi": "Sensei's Forbidden Scrolls",
                    "Techno Exchange": "Model Coefficients & Details",
                    "Imperial Wealth Club": "Analyst's Ledger"
                }[theme]):
                    for model_type, model in models.items():
                        st.write(f"**{model_type}**")
                        if model_type in ["Linear Regression", "Logistic Regression"]:
                            st.dataframe(pd.DataFrame({
                                'Feature': ['Intercept'] + st.session_state.pipeline['features'],
                                'Coefficient': [model.intercept_] + list(model.coef_.flatten())
                            }))
                            st.markdown(f"""
                                <div class="interpretation">
                                { {
                                    "Financial Shinobi": "Power seals show each jutsu's impact. Positive seals boost the target, negative seals weaken it. Greater seals wield stronger influence.",
                                    "Techno Exchange": "Coefficients show each feature's impact. Positive values increase the target, negative values decrease it.",
                                    "Imperial Wealth Club": "Coefficients show each indicator's effect. Positive values increase the entry, negative values decrease it."
                                }[theme] }
                                </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.dataframe(pd.DataFrame(model.cluster_centers_, columns=st.session_state.pipeline['features']))
                            st.markdown(f"""
                                <div class="interpretation">
                                { {
                                    "Financial Shinobi": "Clan centers reveal average jutsu values per group. Compare centers to see clan differences (e.g., high vs. low volatility).",
                                    "Techno Exchange": "Cluster centers show average feature values per group. Compare centers to see group differences.",
                                    "Imperial Wealth Club": "Group centers show average indicator values per group. Compare centers to see group differences."
                                }[theme] }
                                </div>
                            """, unsafe_allow_html=True)
                if st.button(next_btn, key="train_next"):
                    st.session_state.pipeline['current_step'] = 6
                    st.rerun()
            except Exception as e:
                st.error({
                    "Financial Shinobi": f"❌ Sensei training failed: {e}",
                    "Techno Exchange": f"❌ Model training failed: {e}",
                    "Imperial Wealth Club": f"❌ Analyst training failed: {e}"
                }[theme])

def evaluation_step():
    theme = st.session_state.theme
    header, next_btn = THEME_STEP_LABELS["evaluate"][theme]
    st.header(header)
    if not st.session_state.pipeline['model_trained']:
        st.warning({
            "Financial Shinobi": "Master sensei first!",
            "Techno Exchange": "Train model first!",
            "Imperial Wealth Club": "Train analyst first!"
        }[theme])
        return
    models, X_test, y_test = (st.session_state.pipeline[k] for k in ['models', 'X_test', 'y_test'])
    try:
        y_preds = {mt: m.predict(X_test) for mt, m in models.items()}
        st.session_state.pipeline['y_preds'] = y_preds
        metrics_df = pd.DataFrame(columns=['Model', 'RMSE', 'R²'])
        for mt, yp in y_preds.items():
            if mt != "K-Means Clustering":
                mse = mean_squared_error(y_test, yp)
                metrics_df = pd.concat([metrics_df, pd.DataFrame({
                    'Model': [mt], 'RMSE': [np.sqrt(mse)], 'R²': [r2_score(y_test, yp)]
        })], ignore_index=True)
        if not metrics_df.empty:
            st.subheader({
                "Financial Shinobi": "Prophecy Power",
                "Techno Exchange": "Model Performance",
                "Imperial Wealth Club": "Result Strength"
            }[theme])
            st.dataframe(metrics_df.style.format({'RMSE': '{:.4f}', 'R²': '{:.4f}'}), use_container_width=True)
            interp = {
                "Financial Shinobi": "- **RMSE**: Lower seals mean sharper prophecies (less error).\n- **R²**: Closer to 1 means the sensei captures the target's spirit. Negative R² signals a weak prophecy.",
                "Techno Exchange": "- **RMSE**: Lower means better predictions.\n- **R²**: Closer to 1 means the model explains more variance. Negative R² means poor fit.",
                "Imperial Wealth Club": "- **RMSE**: Lower means more accurate results.\n- **R²**: Closer to 1 means the analyst explains more variance. Negative R² means poor fit."
            }[theme]
            st.markdown(f"""
                <div class="interpretation">
                {interp}
                </div>
            """, unsafe_allow_html=True)
        # --- Add this after the metrics table and interpretation block in evaluation_step ---

        # Actual vs Predicted Scatter Plot
        scatter_title = {
            "Financial Shinobi": "Actual vs Prophesied Seals",
            "Techno Exchange": "Actual vs Predicted Values",
            "Imperial Wealth Club": "Actual vs Forecasted Entries"
        }[theme]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_test, y=y_test, mode='lines', name='Perfect Prediction', line=dict(color='#F8F8FF', dash='dash')))
        for mt, yp in y_preds.items():
            if mt != "K-Means Clustering":
                fig.add_trace(go.Scatter(x=y_test, y=yp, mode='markers', name=f'{mt} Predictions', marker=dict(size=8, opacity=0.7, color='#39FF14')))
        plot_config(fig, scatter_title, "Actual", "Predicted", 600, 400)
        st.plotly_chart(fig)
        interp = {
            "Financial Shinobi": "Seals near the sacred line are true prophecies. Scattered seals reveal errors. Hover to compare actual vs. prophesied seals.",
            "Techno Exchange": "Points near the line are accurate predictions. Scatter indicates error. Hover for details.",
            "Imperial Wealth Club": "Entries near the line are accurate forecasts. Scatter shows error. Hover for details."
        }[theme]
        st.markdown(f"<div class=\"interpretation\">{interp}</div>", unsafe_allow_html=True)

        # Residuals Plot
        for mt, yp in y_preds.items():
            if mt != "K-Means Clustering":
                residual_df = pd.DataFrame({'Predicted': yp, 'Residuals': y_test - yp})
                res_title = {
                    "Financial Shinobi": f"Residual Clash - {mt}",
                    "Techno Exchange": f"Residual Plot - {mt}",
                    "Imperial Wealth Club": f"Residual Ledger - {mt}"
                }[theme]
                fig = px.scatter(residual_df, x='Predicted', y='Residuals', title=res_title, width=600, height=400, color_discrete_sequence=['#39FF14'])
                fig.add_hline(y=0, line_dash="dash", line_color="#F8F8FF")
                plot_config(fig, res_title, "Predicted", "Residuals", 600, 400)
                st.plotly_chart(fig)
                interp = {
                    "Financial Shinobi": "Residuals (prophecy errors) should scatter like blood drops around zero. Patterns suggest missed trends.",
                    "Techno Exchange": "Residuals should be randomly scattered around zero. Patterns may indicate bias.",
                    "Imperial Wealth Club": "Residuals should cluster around zero. Patterns may indicate systematic error."
                }[theme]
                st.markdown(f"<div class=\"interpretation\">{interp}</div>", unsafe_allow_html=True)

        # Histogram of Residuals
        for mt, yp in y_preds.items():
            if mt != "K-Means Clustering":
                residual_df = pd.DataFrame({'Residuals': y_test - yp})
                hist_title = {
                    "Financial Shinobi": f"Prophecy Error Storm - {mt}",
                    "Techno Exchange": f"Residual Distribution - {mt}",
                    "Imperial Wealth Club": f"Error Distribution - {mt}"
                }[theme]
                fig = px.histogram(residual_df, x='Residuals', title=hist_title, nbins=30, color_discrete_sequence=['#39FF14'], opacity=0.7)
                plot_config(fig, hist_title, "Residuals", "Count", 600, 400)
                st.plotly_chart(fig)
                interp = {
                    "Financial Shinobi": "A storm peaking near zero suggests an unbiased sensei. Skewed or wild storms signal systematic errors.",
                    "Techno Exchange": "A peak near zero means unbiased model. Skewed or wide distribution signals error.",
                    "Imperial Wealth Club": "A peak near zero means accurate analyst. Skewed or wide distribution signals error."
                }[theme]
                st.markdown(f"<div class=\"interpretation\">{interp}</div>", unsafe_allow_html=True)
        # ... (continue this pattern for all graphs and interpretation blocks in this step) ...
        if st.button(next_btn, key="evaluation_next"):
            st.session_state.pipeline['model_evaluated'] = True
            st.session_state.pipeline['current_step'] = 7
            st.rerun()
    except Exception as e:
        st.error({
            "Financial Shinobi": f"❌ Prophecy unleash failed: {e}",
            "Techno Exchange": f"❌ Evaluation failed: {e}",
            "Imperial Wealth Club": f"❌ Review failed: {e}"
        }[theme])
        

def results_visualization_step():
    theme = st.session_state.theme
    header = THEME_STEP_LABELS["results"][theme][0]
    st.header(header)
    if not st.session_state.pipeline['model_evaluated']:
        st.warning({
            "Financial Shinobi": "Unleash prophecies first!",
            "Techno Exchange": "Evaluate model first!",
            "Imperial Wealth Club": "Review results first!"
        }[theme])
        return

    y_test = st.session_state.pipeline['y_test']
    y_preds = st.session_state.pipeline['y_preds']
    models = st.session_state.pipeline['models']

    # Final comparison plot: Actual vs All Model Predictions
    comp_title = {
        "Financial Shinobi": "Final Prophecy Showdown",
        "Techno Exchange": "Final Model Comparison",
        "Imperial Wealth Club": "Ledger Forecast Comparison"
    }[theme]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_test, y=y_test, mode='lines', name={
        "Financial Shinobi": "Sacred Seal",
        "Techno Exchange": "Perfect Prediction",
        "Imperial Wealth Club": "True Ledger"
    }[theme], line=dict(color='#F8F8FF', dash='dash')))
    for mt, yp in y_preds.items():
        if mt != "K-Means Clustering":
            fig.add_trace(go.Scatter(x=y_test, y=yp, mode='markers', name=f'{mt}', marker=dict(size=8, opacity=0.7, color='#39FF14')))
    plot_config(fig, comp_title, {
        "Financial Shinobi": "Actual Seal",
        "Techno Exchange": "Actual Value",
        "Imperial Wealth Club": "Actual Entry"
    }[theme], {
        "Financial Shinobi": "Prophesied Seal",
        "Techno Exchange": "Predicted Value",
        "Imperial Wealth Club": "Forecasted Entry"
    }[theme], 700, 500)
    st.plotly_chart(fig)
    
    # Download buttons for each model's predictions
    for mt, yp in y_preds.items():
        if mt != "K-Means Clustering":
            results_df = pd.DataFrame({
                {
                    "Financial Shinobi": "Actual Seal",
                    "Techno Exchange": "Actual Value",
                    "Imperial Wealth Club": "Actual Entry"
                }[theme]: y_test,
                {
                    "Financial Shinobi": f"Prophesied Seal ({mt})",
                    "Techno Exchange": f"Predicted Value ({mt})",
                    "Imperial Wealth Club": f"Forecasted Entry ({mt})"
                }[theme]: yp
            })
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label={
                    "Financial Shinobi": f"📥 Download Prophecy ({mt})",
                    "Techno Exchange": f"⬇️ Download Predictions ({mt})",
                    "Imperial Wealth Club": f"💾 Download Ledger Results ({mt})"
                }[theme],
                data=csv,
                file_name=f"{mt.replace(' ', '_').lower()}_results.csv",
                mime='text/csv',
                key=f"download_{mt}"
            )

    # Final interpretation/conclusion block
    interp = {
        "Financial Shinobi": "<b>Your shinobi journey is complete!</b> The forbidden scrolls are revealed. Download your prophecies and share your legend with the clan. <br><br> <i>May your seals always be sharp and your jutsu ever stronger!</i>",
        "Techno Exchange": "<b>Analysis complete!</b> Download your predictions and use these insights to inform your next trades. <br><br> <i>May your portfolio always be in the green!</i>",
        "Imperial Wealth Club": "<b>The ledger is closed.</b> Download your results and reflect on your financial wisdom. <br><br> <i>May your accounts always balance in your favor!</i>"
    }[theme]
    st.markdown(f"<div class=\"interpretation\">{interp}</div>", unsafe_allow_html=True)

def landing_page():
    theme_options = [
        {
            "key": "Financial Shinobi",
            "icon": "⚔️",
            "name": "Financial Shinobi",
            "desc": "Anime-inspired, dark, energetic, and bold."
        },
        {
            "key": "Techno Exchange",
            "icon": "🟦",
            "name": "Techno Exchange Protocol",
            "desc": "Futuristic, neon, cyber/AI, and modern."
        },
        {
            "key": "Imperial Wealth Club",
            "icon": "👑",
            "name": "Imperial Wealth Club",
            "desc": "Classic, gold, vintage, and elegant."
        }
    ]
    st.markdown("""
        <style>
        body, .stApp {
            background: linear-gradient(120deg, #f5f7fa 0%, #c3cfe2 100%) !important;
        }
        .landing-hero {
            text-align: center;
            margin-top: 60px;
            margin-bottom: 30px;
        }
        .landing-title {
            font-size: 3rem;
            font-weight: 800;
            letter-spacing: 2px;
            margin-bottom: 0.5rem;
            color: #222;
        }
        .landing-tagline {
            font-size: 1.25rem;
            color: #555;
            margin-bottom: 2.5rem;
        }
        .theme-card {
            background: #fff;
            border-radius: 18px;
            box-shadow: 0 2px 16px 0 rgba(0,0,0,0.08);
            border: 2.5px solid #e0e7ef;
            padding: 1.5rem 1.2rem 1.2rem 1.2rem;
            text-align: center;
            transition: box-shadow 0.2s, border 0.2s, transform 0.15s;
            cursor: pointer;
            margin-bottom: 0px;
            position: relative;
            min-height: 370px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .theme-card:hover {
            border: 2.5px solid #0078fa;
            box-shadow: 0 8px 32px 0 #0078fa33;
            transform: scale(1.045);
            z-index: 2;
        }
        .theme-icon {
            font-size: 2.7rem;
            margin-bottom: 0.7rem;
            margin-top: 0.2rem;
            filter: drop-shadow(0 2px 8px #e0e7ef);
        }
        .theme-img {
            width: 90px;
            height: 90px;
            object-fit: cover;
            border-radius: 12px;
            margin-bottom: 0.7rem;
            box-shadow: 0 1px 8px #eee;
            border: 2px solid #e0e7ef;
            background: #f8fafc;
        }
        .theme-name {
            font-size: 1.35rem;
            font-weight: 700;
            margin-bottom: 0.3rem;
            color: #222;
            letter-spacing: 0.5px;
        }
        .theme-desc {
            font-size: 1rem;
            color: #666;
            margin-bottom: 1.1rem;
            min-height: 48px;
        }
        .theme-btn-overlap .stButton > button {
            margin-top: -18px !important;
            box-shadow: 0 2px 16px 0 #0078fa33;
        }
        .stButton > button {
            font-size: 1.08rem;
            font-weight: 700;
            padding: 12px 36px;
            border-radius: 8px;
            background: linear-gradient(90deg, #0078fa, #00c6fb);
            color: #fff;
            border: none;
            box-shadow: 0 2px 16px 0 #0078fa33;
            transition: background 0.2s, box-shadow 0.2s;
            cursor: pointer;
            width: 90%;
            margin-left: auto;
            margin-right: auto;
        }
        .stButton > button:hover {
            background: linear-gradient(90deg, #00c6fb, #0078fa);
            box-shadow: 0 4px 24px 0 #00c6fb33;
        }
        </style>
    """, unsafe_allow_html=True)
    st.markdown('<div class="landing-hero">'
                '<div class="landing-title">Market Master</div>'
                '<div class="landing-tagline">Choose your financial adventure!</div>'
                '</div>', unsafe_allow_html=True)

    cols = st.columns(len(theme_options))
    for idx, theme in enumerate(theme_options):
        with cols[idx]:
            st.markdown(f"""
                <div class='theme-card'>
                    <div class='theme-icon'>{theme['icon']}</div>
                    <img class='theme-img' src='{THEME_GIFS[theme['key']]}' />
                    <div class='theme-name'>{theme['name']}</div>
                    <div class='theme-desc'>{theme['desc']}</div>
                </div>
            """, unsafe_allow_html=True)
            with st.container():
                st.markdown("<div class='theme-btn-overlap'>", unsafe_allow_html=True)
                if st.button(f"Select {theme['name']}", key=f"select_{theme['key']}"):
                    st.session_state.theme = theme['key']
                    st.session_state.landing_done = True
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

# Main app
def main():
    init_session_state()
    if not st.session_state.get("landing_done", False):
        landing_page()
        return

    # Set page config and apply Theme (now that theme is chosen)
    theme = st.session_state.theme
    site_title = THEME_TITLES.get(theme, "Market Master")
    st.markdown(f"""
        <style>
        {get_theme_css()}
        </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.header(THEME_SIDEBAR_TITLES.get(theme, "Training Stages"))
        theme_options = ["Financial Shinobi", "Techno Exchange", "Imperial Wealth Club"]
        selected_theme = st.selectbox("Choose Theme", theme_options, 
                                     index=theme_options.index(st.session_state.theme), key="sidebar_theme")
        if selected_theme != st.session_state.theme:
            st.session_state.theme = selected_theme
            st.rerun()
        st.divider()
        sidebar_steps = THEME_SIDEBAR_STEPS[theme]
        steps = [
            (sidebar_steps[0], 0, "Begin your quest!", None),
            (sidebar_steps[1], 1, "Collect market data", None),
            (sidebar_steps[2], 2, "Clean your data", 'data_loaded'),
            (sidebar_steps[3], 3, "Create powerful features", 'preprocessed'),
            (sidebar_steps[4], 4, "Train/test split", 'features_engineered'),
            (sidebar_steps[5], 5, "Train your model", 'data_split'),
            (sidebar_steps[6], 6, "Test your model", 'model_trained'),
            (sidebar_steps[7], 7, "Visualize outcomes", 'model_evaluated')
        ]
        progress = sum([st.session_state.pipeline.get(c, False) for _, _, _, c in steps if c]) / len([c for _, _, _, c in steps if c]) * 100
        st.markdown(f"<div class='progress-bar'><div class='progress-fill' style='width: {progress}%'></div></div>", unsafe_allow_html=True)
        for name, step, tooltip, condition in steps:
            disabled = False if condition is None else not st.session_state.pipeline.get(condition, False)
            label = f"{name} ⚔️" if condition and st.session_state.pipeline.get(condition, False) else name
            st.button(label, key=f"step_{step}", disabled=disabled, 
                      on_click=lambda s=step: st.session_state.pipeline.update({'current_step': s}), help=tooltip)
        st.divider()
        st.markdown('<div class="center-image"><img src="https://gifdb.com/images/high/anime-money-safe-1989-riding-bean-tlrjh66tg0es3idz.gif" width="220"></div>', unsafe_allow_html=True)
        st.button("🔄 Start New Journey", key="reset", 
                  on_click=lambda: [st.session_state.clear(), init_session_state(), st.session_state.update({'landing_done': False})], 
                  help="Begin a new journey")
    
    step_funcs = [welcome_step, load_data_step, preprocessing_step, feature_engineering_step,
                  train_test_split_step, model_training_step, evaluation_step, results_visualization_step]
    step_funcs[st.session_state.pipeline['current_step']]()

if __name__ == "__main__":
    main()