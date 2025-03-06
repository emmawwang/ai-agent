import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # Base directory
    BASE_DIR = Path(__file__).parent
    
    # Data directory
    DATA_DIR = BASE_DIR / "data"
    
    # Database
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///workforce_data.db")
    
    # API Keys
    API_KEYS = {
        "BLS_API_KEY": os.getenv("BLS_API_KEY", ""),
        "ONET_API_KEY": os.getenv("ONET_API_KEY", ""),
        "LINKEDIN_API_KEY": os.getenv("LINKEDIN_API_KEY", ""),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "PERPLEXITY_API_KEY": os.getenv("PERPLEXITY_API_KEY")
    }
    
    # Project paths
    DATA_DIR = BASE_DIR / "data_cache"
    LOGS_DIR = BASE_DIR / "logs"
    
    # Data update frequency (in hours)
    UPDATE_FREQUENCY = {
        "linkedin": 24,
        "bls": 168,  # Weekly
        "burning_glass": 24,
        "onet": 168  # Weekly
    }
    
    # AI Impact scoring weights
    AI_IMPACT_WEIGHTS = {
        "automation_risk": 0.4,
        "ai_adoption_rate": 0.3,
        "skill_relevance": 0.3
    }

# File paths
ECONOMIC_DATA_FILE = "insights/data/economic_data.csv"
AUTOMATION_FILE = "data/EconomicIndex/automation_vs_augmentation.csv"
BLS_EMPLOYMENT_FILE = "insights/data/bls_employment_projection_2023_33_annualized_rounded.csv"
TASK_MAPPINGS_FILE = "data/EconomicIndex/onet_task_mappings.csv"
TASK_STATEMENTS_FILE = "data/EconomicIndex/onet_task_statements.csv"
ONET_TASK_MAPPINGS_FILE = "insights/data/onet_task_mappings.csv"

# Thresholds
HIGH_GROWTH_THRESHOLD = 0.08  # 8%
MODERATE_GROWTH_THRESHOLD = 0.04  # 4%
GROWING_INDUSTRY_THRESHOLD = 0.3
STABLE_INDUSTRY_THRESHOLD = 0.6

# Create a basic ONET task mappings file if it doesn't exist
if not os.path.exists(ONET_TASK_MAPPINGS_FILE):
    import pandas as pd
    
    # Create basic task data
    task_data = {
        'task_name': [
            'Software development and coding',
            'Data analysis and reporting',
            'Project management',
            'Customer communication',
            'Documentation'
        ],
        'claude_usage_pct': [0.027, 0.023, 0.019, 0.015, 0.012]
    }
    
    # Save to CSV
    df = pd.DataFrame(task_data)
    df.to_csv(ONET_TASK_MAPPINGS_FILE, index=False) 