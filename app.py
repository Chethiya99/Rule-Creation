import streamlit as st
import pandas as pd
import groq
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import os

# Initialize Groq client
try:
    client = groq.Client(api_key=st.secrets["GROQ_API_KEY"])
except Exception as e:
    st.error(f"Failed to initialize Groq client: {str(e)}")
    st.stop()

# Configuration
CSV_DIR = "data"  # Directory containing CSV files

@st.cache_data
def get_csv_structures() -> Dict[str, List[str]]:
    """Get all CSV files and their columns from the data directory"""
    csv_structures = {}
    
    # Create directory if it doesn't exist
    if not os.path.exists(CSV_DIR):
        os.makedirs(CSV_DIR)
        st.warning(f"Created directory {CSV_DIR}. Please add your CSV files there.")
        return csv_structures
    
    # Read each CSV file and get columns
    for filename in os.listdir(CSV_DIR):
        if filename.endswith(".csv"):
            try:
                # Read just the header to get column names
                filepath = os.path.join(CSV_DIR, filename)
                df = pd.read_csv(filepath, nrows=0)
                csv_structures[filename] = list(df.columns)
            except Exception as e:
                st.warning(f"Could not read columns from {filename}: {str(e)}")
    
    return csv_structures

# Get CSV structures when app loads
CSV_STRUCTURES = get_csv_structures()

def clean_user_input(text: str) -> str:
    """Clean user input by removing extra spaces between characters"""
    text = ' '.join(text.split())
    return text.replace(" o n ", " on ").replace(" A N D ", " AND ").replace(" O R ", " OR ")

def generate_prompt_guidance(user_input: str, modification_request: Optional[str] = None) -> str:
    """Generate guidance for the AI based on user input and available data"""
    # Format available data for the prompt
    available_data = []
    for filename, columns in CSV_STRUCTURES.items():
        available_data.append(f"Data source: {filename}")
        available_data.append(f"Columns: {', '.join(columns)}")
        available_data.append("")  # Empty line for readability
    
    base_prompt = f"""
    You are a financial rule generation assistant. Your task is to help create rules for mortgage holders based on EXACT column names from available data sources.

    AVAILABLE DATA SOURCES AND THEIR EXACT COLUMN NAMES:
    {('\n').join(available_data)}

    USER REQUIREMENT: "{user_input}"

    CRITICAL INSTRUCTIONS:
    1. You MUST use ONLY the exact column names shown above
    2. You MUST use ONLY the data sources shown above
    3. Map user concepts to exact column names (e.g., if user says "payment amount", use "payment_amount" if that exists)
    4. For dates, use exact column names like "payment_date" or "start_date"
    5. For amounts, use exact column names like "loan_amount" or "payment_amount"
    6. Never invent or make up column names - they must exist exactly as shown above
    """
    
    if modification_request:
        base_prompt += f"\nUSER REQUESTED MODIFICATIONS: {modification_request}"
    
    base_prompt += """
    YOUR TASKS:
    1. Identify which exact data sources are needed
    2. Select which exact columns from each source should be used
    3. Create a logical rule structure with proper AND/OR conditions
    4. Output in this exact JSON format:
    {
        "rules": [
            {
                "id": "generated_id",
                "dataSource": "exact_filename.csv",
                "field": "exact_column_name",
                "eligibilityPeriod": "N/A" or "Rolling X days",
                "function": "N/A" or aggregation function,
                "operator": "comparison_operator",
                "value": "comparison_value",
                "priority": null,
                "ruleType": "condition" or "conditionGroup",
                "connector": "AND" or "OR" or null,
                "conditions": [ /* for conditionGroup only */ ]
            }
        ]
    }

    RESPONSE REQUIREMENTS:
    1. Output ONLY the JSON
    2. Use ONLY the exact data sources and column names provided
    3. No additional explanation or commentary
    """
    
    return base_prompt

def generate_rule_with_llama(user_input: str, modification_request: Optional[str] = None) -> Dict[str, Any]:
    """Use Groq/Llama to generate a rule based on user input"""
    prompt = generate_prompt_guidance(user_input, modification_request)
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a financial rule generator that uses ONLY exact column names from provided data sources."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama3-70b-8192",
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        response_content = chat_completion.choices[0].message.content
        
        # Parse and validate the response
        rule = json.loads(response_content)
        
        # Validate all fields use real column names
        if "rules" in rule:
            for rule_item in rule["rules"]:
                data_source = rule_item.get("dataSource")
                field = rule_item.get("field")
                
                # Validate data source exists
                if data_source not in CSV_STRUCTURES:
                    st.error(f"Invalid data source in generated rule: {data_source}")
                    return None
                
                # Validate field exists in data source
                if field and field not in CSV_STRUCTURES[data_source]:
                    st.error(f"Invalid field '{field}' for data source '{data_source}'")
                    return None
                
                # Validate condition groups
                for condition in rule_item.get("conditions", []):
                    ds = condition.get("dataSource")
                    fld = condition.get("field")
                    
                    if ds not in CSV_STRUCTURES:
                        st.error(f"Invalid data source in condition group: {ds}")
                        return None
                    
                    if fld and fld not in CSV_STRUCTURES[ds]:
                        st.error(f"Invalid field '{fld}' for data source '{ds}'")
                        return None
        
        return rule
    
    except json.JSONDecodeError:
        st.error("Failed to parse AI response as JSON")
        return None
    except Exception as e:
        st.error(f"Error generating rule: {str(e)}")
        return None

# [Rest of your existing functions remain exactly the same...]
# display_rule_ui()
# initialize_session_state()
# display_chat_message()
# handle_user_confirmation()
# generate_new_rule()

def main():
    st.set_page_config(page_title="Mortgage Rule Generator", layout="wide")
    st.title("🏦 Mortgage Rule Generator with Llama 3")
    
    # Custom CSS
    st.markdown("""
    <style>
        .stChatFloatingInputContainer { bottom: 20px; }
        .stChatMessage { padding: 12px; border-radius: 8px; margin-bottom: 12px; }
        .assistant-message { background-color: #f0f2f6; }
        .user-message { background-color: #e3f2fd; }
        .stTextInput input, .stSelectbox select { font-size: 14px !important; }
        .stExpander { margin-bottom: 15px; border: 1px solid #e0e0e0; border-radius: 8px; }
        .stExpander .streamlit-expanderHeader { font-weight: bold; background-color: #f5f5f5; padding: 10px 15px; }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    initialize_session_state()
    
    # Show available data sources in sidebar
    with st.sidebar:
        st.subheader("Available Data Sources")
        if not CSV_STRUCTURES:
            st.warning("No CSV files found in data directory")
        else:
            for filename, columns in CSV_STRUCTURES.items():
                with st.expander(filename):
                    st.write("Columns:", ", ".join(columns))
    
    # Main layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.session_state.current_rule:
            display_rule_ui(st.session_state.current_rule)
            
            if st.session_state.confirmed:
                st.success("✅ Final Rule Confirmed")
                st.json(st.session_state.current_rule)
                
                json_str = json.dumps(st.session_state.current_rule, indent=2)
                st.download_button(
                    label="Download Rule JSON",
                    data=json_str,
                    file_name=f"mortgage_rule_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
                
                if st.button("Create New Rule"):
                    st.session_state.messages = [
                        {"role": "assistant", "content": "Let's create a new rule. What criteria would you like to use?"}
                    ]
                    st.session_state.current_rule = None
                    st.session_state.confirmed = False
                    st.session_state.user_prompt = ""
                    st.rerun()
    
    with col2:
        st.subheader("Rule Assistant")
        
        for message in st.session_state.messages:
            display_chat_message(message["role"], message["content"])
        
        if prompt := st.chat_input("Type your message here..."):
            cleaned_prompt = clean_user_input(prompt)
            st.session_state.messages.append({"role": "user", "content": cleaned_prompt})
            display_chat_message("user", cleaned_prompt)
            
            if not st.session_state.user_prompt:
                st.session_state.user_prompt = cleaned_prompt
                generate_new_rule()
                st.rerun()
            elif st.session_state.awaiting_confirmation:
                if "yes" in cleaned_prompt.lower() or "correct" in cleaned_prompt.lower():
                    handle_user_confirmation(True)
                else:
                    handle_user_confirmation(False)
                st.rerun()
            elif st.session_state.awaiting_modification:
                generate_new_rule()
                st.rerun()
            else:
                st.session_state.user_prompt = cleaned_prompt
                st.session_state.current_rule = None
                st.session_state.confirmed = False
                generate_new_rule()
                st.rerun()

if __name__ == "__main__":
    main()
