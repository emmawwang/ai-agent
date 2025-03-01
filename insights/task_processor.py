import re

def extract_tasks(text):
    """Extract individual tasks from job description with improved parsing"""
    tasks = []
    
    # First try to find bullet points, dashes, or numbered lists
    bullet_pattern = r'(?:^|\n)\s*(?:•|\*|-|\d+\.)\s*(.*?)(?=(?:^|\n)\s*(?:•|\*|-|\d+\.)|$)'
    bullet_tasks = re.findall(bullet_pattern, text, re.MULTILINE | re.DOTALL)
    
    for task in bullet_tasks:
        task = task.strip()
        if task and len(task) > 10:
            # Remove any sub-bullets
            task = re.sub(r'(?:^|\n)\s*(?:[○◦→–—]|\([a-z]\))\s*.*?(?=(?:^|\n)|$)', '', task, flags=re.MULTILINE | re.DOTALL)
            tasks.append(task)
    
    # If few or no bullet points found, try to split by sentences
    if len(tasks) < 3:
        # Look for sections like "Responsibilities:" or "Requirements:"
        sections = re.findall(r'(?:Responsibilities|Requirements|Qualifications|Skills|Duties):?\s*(.*?)(?=(?:^|\n)\s*[A-Z][a-z]+:|\Z)', text, re.MULTILINE | re.DOTALL | re.IGNORECASE)
        
        for section in sections:
            # Split by sentences and find ones with action verbs
            sentences = re.split(r'(?<=[.!?])\s+', section)
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence and len(sentence) > 15 and re.search(r'\b(?:manage|develop|create|analyze|implement|design|coordinate|maintain|support|provide|prepare|conduct|perform|ensure|establish|evaluate|communicate|build|work|code|program|test|write|review|collaborate|research|lead|deliver|organize|solve|improve|optimize)\b', sentence.lower()):
                    tasks.append(sentence)
    
    # If still not enough tasks, try more aggressive extraction
    if len(tasks) < 2:
        # Get any sentence with a verb
        sentences = re.split(r'(?<=[.!?])\s+', text)
        for sentence in sentences:
            sentence = sentence.strip()
            # Check for verbs and minimum length
            if sentence and len(sentence) > 20 and re.search(r'\b(?:is|are|was|were|will|have|has|had|do|does|did|can|could|may|might|shall|should|would|make|create|provide|work|develop|manage|lead|build|design|implement|analyze|support|maintain|improve)\b', sentence.lower()):
                if sentence not in tasks:  # Avoid duplicates
                    tasks.append(sentence)
    
    # Print what we found for debugging
    print(f"Extracted task details: {len(tasks)} tasks found")
    
    return tasks

def preprocess_text(text):
    """Basic text preprocessing"""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase and remove punctuation
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def detect_industry(text):
    """Detect the industry from a job description using keyword analysis"""
    text_lower = text.lower()
    
    # Industry keywords mapping with expanded categories
    industry_keywords = {
        "Technology": ["software", "developer", "coding", "programming", "application", "web", "app", 
                       "database", "cloud", "network", "it ", "information technology", "computer", 
                       "system", "security", "data science", "artificial intelligence", "machine learning",
                       "infrastructure", "devops", "full-stack", "frontend", "backend", "api", "cybersecurity"],
        
        "Healthcare": ["patient", "medical", "healthcare", "doctor", "nurse", "physician", "clinical", 
                       "hospital", "health", "care", "treatment", "therapy", "diagnostic", "pharma",
                       "pharmacy", "medicine", "telehealth", "biotech", "caregiver", "elderly"],
        
        "Finance": ["finance", "financial", "banking", "investment", "accounting", "bank", "loan", 
                   "credit", "money", "asset", "fund", "trading", "portfolio", "wealth", "fintech",
                   "insurance", "mortgage", "broker", "stocks", "securities", "compliance"],
        
        "Education": ["education", "school", "teacher", "teaching", "student", "learning", "academic", 
                    "curriculum", "classroom", "instruction", "college", "university", "professor",
                    "tutor", "course", "degree", "training", "educational", "e-learning"],
        
        "Retail": ["retail", "store", "sales", "customer", "product", "merchandise", "inventory", 
                  "consumer", "sell", "shop", "purchase", "buyer", "brand", "commerce", "e-commerce",
                  "market", "shopper", "checkout", "pos", "point of sale"],
        
        "Manufacturing": ["manufacturing", "production", "assembly", "factory", "fabrication", 
                        "quality control", "machine", "equipment", "industrial", "plant", "warehouse",
                        "supply chain", "logistics", "distribution", "inventory", "raw material"],
        
        "Marketing": ["marketing", "advertisement", "campaign", "brand", "social media", "content", 
                     "market research", "seo", "digital marketing", "promotion", "advertising",
                     "public relations", "pr", "branding", "audience", "influencer", "creative"],
        
        "Legal": ["legal", "law", "attorney", "lawyer", "compliance", "regulation", "contract", 
                "litigation", "paralegal", "counsel", "firm", "legal aid", "judiciary",
                "court", "statute", "legislation", "regulatory"],
        
        "Hospitality": ["hospitality", "hotel", "restaurant", "guest", "food", "beverage", "tourism", 
                      "travel", "accommodation", "service", "catering", "chef", "culinary", "kitchen",
                      "housekeeping", "lodging", "booking", "reservation"],
        
        "Food Industry": ["food", "beverage", "restaurant", "culinary", "chef", "kitchen", "catering",
                         "menu", "cooking", "baking", "pastry", "nutrition", "dietary", "ingredient",
                         "recipe", "grocery", "farm", "agriculture", "produce", "organic", "sustainable"],
        
        "AI and Data": ["artificial intelligence", "machine learning", "data science", "deep learning",
                        "neural network", "algorithm", "big data", "analytics", "data mining", "nlp",
                        "natural language processing", "computer vision", "predictive modeling", "reinforcement learning"],
                      
        "Construction": ["construction", "building", "architect", "engineering", "contractor", "project manager",
                        "site", "worker", "safety", "blueprint", "permit", "inspection", "renovation",
                        "commercial", "residential", "infrastructure", "real estate", "property"],
    }
    
    # Count keyword occurrences for each industry
    industry_scores = {}
    for industry, keywords in industry_keywords.items():
        score = sum(1 for keyword in keywords if keyword in text_lower)
        industry_scores[industry] = score
    
    # Get the industry with the highest score
    if max(industry_scores.values()) > 0:
        return max(industry_scores.items(), key=lambda x: x[1])[0]
    else:
        return "General"