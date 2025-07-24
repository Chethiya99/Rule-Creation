import streamlit as st
import pandas as pd
import groq
import json
from typing import List, Dict, Any, Optional
import time
from datetime import datetime
import os

# Initialize Groq client
try:
    client = groq.Client(api_key=st.secrets["GROQ_API_KEY"])
except Exception as e:
    st.error(f"Failed to initialize Groq client: {str(e)}")
    st.stop()

# Load actual CSV files and extract column names
def load_csv_structures():
    csv_structures = {}
    data_dir = "data"  # Directory containing your CSV files
    
    try:
        # List of your actual CSV files
        csv_files = [
            "sample_mortgage_accounts.csv",
            "sample_loan_repayments.csv",
            "sample_telco_billing.csv",
            "sample_product_enrollments.csv",
            "sample_customer_profiles.csv",
            "sample_savings_account_transactions.csv",
            "sample_credit_card_transactions.csv"
        ]
        
        for file in csv_files:
            file_path = os.path.join(data_dir, file)
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    csv_structures[file] = list(df.columns)
                    st.session_state[f"df_{file}"] = df  # Store dataframe in session state
                except Exception as e:
                    st.warning(f"Error reading {file_path}: {str(e)}")
                    # Fallback to sample structure if file can't be read
                    if file == "sample_mortgage_accounts.csv":
                        csv_structures[file] = ["customer_id", "account_status", "loan_amount", "interest_rate", "start_date", "term"]
                    elif file == "sample_credit_card_transactions.csv":
                        csv_structures[file] = ["transaction_id", "card_id", "customer_id", "amount", "date", "merchant", "category"]
            else:
                st.warning(f"File not found: {file_path}")
                # Fallback to sample structure if file not found
                if file == "sample_mortgage_accounts.csv":
                    csv_structures[file] = ["customer_id", "account_status", "loan_amount", "interest_rate", "start_date", "term"]
                elif file == "sample_credit_card_transactions.csv":
                    csv_structures[file] = ["transaction_id", "card_id", "customer_id", "amount", "date", "merchant", "category"]
        
        return csv_structures
    
    except Exception as e:
        st.error(f"Error loading CSV structures: {str(e)}")
        return {}

# Load actual CSV structures
CSV_STRUCTURES = load_csv_structures()

def clean_user_input(text: str) -> str:
    """Clean user input by removing extra spaces between characters"""
    # First remove any existing extra spaces
    text = ' '.join(text.split())
    # Handle special cases where spaces might have been added between letters
    return text.replace(" o n ", " on ").replace(" A N D ", " AND ").replace(" O R ", " OR ")

def generate_prompt_guidance(user_input: str, modification_request: Optional[str] = None) -> str:
    """Generate guidance for the AI with actual column names"""
    available_data = "\n".join([f"- {f}: {', '.join(cols)}" for f, cols in CSV_STRUCTURES.items()])
    
    base_prompt = f"""
    You are a financial rule generation assistant. Your task is to help create rules for mortgage holders based on available data sources.

    Available data sources and their ACTUAL columns:
    {available_data}

    The user has provided this requirement: "{user_input}"
    """
    
    if modification_request:
        base_prompt += f"\nThe user requested these modifications: {modification_request}"
    
    base_prompt += """
    Analyze this requirement and:
    1. Carefully select the MOST RELEVANT data sources and EXACT column names from the available options above
    2. Match user's terminology to actual columns (e.g., "spend" -> "amount", "balance" -> "loan_amount")
    3. For time periods, use the closest matching date/duration column
    4. Include all these fields for each condition:
       - dataSource (EXACT filename from available sources)
       - field (EXACT column name from the selected dataSource)
       - eligibilityPeriod (use "Rolling 30 days" for time-based conditions, otherwise "N/A")
       - function (use "sum", "count", "avg" where appropriate, otherwise "N/A")
       - operator
       - value
    5. Output the rule in JSON format matching this schema:
        {
            "rules": [
                {
                    "id": "generated_id",
                    "dataSource": "source_name.csv",
                    "field": "exact_column_name",
                    "eligibilityPeriod": "time_period or N/A",
                    "function": "aggregation_function or N/A",
                    "operator": "comparison_operator",
                    "value": "comparison_value",
                    "priority": null,
                    "ruleType": "condition" or "conditionGroup",
                    "connector": "AND" or "OR" or null,
                    "conditions": [ /* for conditionGroup only */ ]
                }
            ]
        }

    Respond ONLY with the JSON output. Do not include any additional explanation or markdown formatting.
    IMPORTANT: Use ONLY the exact column names and data sources provided above.
    """
    
    return base_prompt

def generate_rule_with_llama(user_input: str, modification_request: Optional[str] = None) -> Dict[str, Any]:
    """Use Groq/Llama to generate a rule with accurate column mapping"""
    prompt = generate_prompt_guidance(user_input, modification_request)
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a financial rule generation expert that creates precise JSON rules using EXACT column names from provided data sources."
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
        
        # Clean the response to extract just the JSON
        try:
            rule = json.loads(response_content)
            
            # Validate that all columns exist in the data sources
            if not validate_rule_columns(rule):
                st.error("Generated rule contains invalid columns. Please try again.")
                return None
                
            return rule
        except json.JSONDecodeError:
            st.error("Failed to parse the AI response. Please try again.")
            return None
    
    except Exception as e:
        st.error(f"Error generating rule: {str(e)}")
        return None

def validate_rule_columns(rule: Dict[str, Any]) -> bool:
    """Validate that all columns in the rule exist in the data sources"""
    if not rule or "rules" not in rule:
        return False
    
    for rule_item in rule["rules"]:
        data_source = rule_item.get("dataSource")
        field = rule_item.get("field")
        
        # Check if data source exists
        if data_source not in CSV_STRUCTURES:
            st.error(f"Invalid data source: {data_source}")
            return False
            
        # Check if field exists in data source
        if field and field not in CSV_STRUCTURES[data_source]:
            st.error(f"Invalid field '{field}' for data source {data_source}")
            return False
            
        # Validate condition groups
        if rule_item.get("ruleType") == "conditionGroup":
            for condition in rule_item.get("conditions", []):
                ds = condition.get("dataSource")
                fld = condition.get("field")
                
                if ds not in CSV_STRUCTURES:
                    st.error(f"Invalid data source in condition group: {ds}")
                    return False
                    
                if fld and fld not in CSV_STRUCTURES[ds]:
                    st.error(f"Invalid field '{fld}' in condition group for data source {ds}")
                    return False
                    
    return True

def display_rule_ui(rule: Dict[str, Any]) -> None:
    """Display the rule in the UI with all required fields"""
    if not rule or "rules" not in rule:
        st.warning("No valid rule generated yet")
        return
    
    st.subheader("Rule Conditions")
    st.markdown("Define the logical conditions for this rule to apply.")
    
    # Priority checkbox
    st.checkbox("Enable priority order and drag & drop", value=False, key="priority_order")
    
    # Main rule display
    for i, rule_item in enumerate(rule["rules"]):
        if rule_item.get("ruleType") == "condition":
            with st.expander(f"Condition {i+1}", expanded=True):
                cols = st.columns(7)
                with cols[0]:
                    selected_ds = st.selectbox(
                        "Data Source",
                        options=list(CSV_STRUCTURES.keys()),
                        index=list(CSV_STRUCTURES.keys()).index(rule_item["dataSource"]) if rule_item["dataSource"] in CSV_STRUCTURES else 0,
                        key=f"ds_{i}"
                    )
                with cols[1]:
                    if selected_ds in CSV_STRUCTURES:
                        st.selectbox(
                            "Field",
                            options=CSV_STRUCTURES[selected_ds],
                            index=CSV_STRUCTURES[selected_ds].index(rule_item["field"]) if rule_item["field"] in CSV_STRUCTURES[selected_ds] else 0,
                            key=f"field_{i}"
                        )
                    else:
                        st.text_input("Field", value=rule_item.get("field", ""), key=f"field_{i}")
                with cols[2]:
                    st.selectbox(
                        "eligibilityPeriod",
                        ["N/A", "Rolling 30 days", "Rolling 60 days", "Rolling 90 days", "Current month"],
                        index=0 if rule_item.get("eligibilityPeriod") == "N/A" else 1,
                        key=f"period_{i}"
                    )
                with cols[3]:
                    st.selectbox(
                        "function",
                        ["N/A", "sum", "count", "avg", "max", "min"],
                        index=0 if rule_item.get("function") == "N/A" else 1,
                        key=f"func_{i}"
                    )
                with cols[4]:
                    st.selectbox(
                        "Operator",
                        ["=", ">", "<", ">=", "<=", "!=", "contains"],
                        index=0,
                        key=f"op_{i}"
                    )
                with cols[5]:
                    st.text_input(
                        "Value",
                        value=rule_item.get("value", ""),
                        key=f"val_{i}"
                    )
                
                if i < len(rule["rules"]) - 1:
                    with cols[6]:
                        st.selectbox(
                            "Connector",
                            ["AND", "OR"],
                            index=0 if rule_item.get("connector", "AND") == "AND" else 1,
                            key=f"conn_{i}"
                        )
        
        elif rule_item.get("ruleType") == "conditionGroup":
            with st.expander(f"Condition Group {i+1}", expanded=True):
                st.markdown("#### Condition Group")
                for j, condition in enumerate(rule_item.get("conditions", [])):
                    cols = st.columns(7)
                    with cols[0]:
                        selected_ds = st.selectbox(
                            "Data Source",
                            options=list(CSV_STRUCTURES.keys()),
                            index=list(CSV_STRUCTURES.keys()).index(condition["dataSource"]) if condition["dataSource"] in CSV_STRUCTURES else 0,
                            key=f"gds_{i}_{j}"
                        )
                    with cols[1]:
                        if selected_ds in CSV_STRUCTURES:
                            st.selectbox(
                                "Field",
                                options=CSV_STRUCTURES[selected_ds],
                                index=CSV_STRUCTURES[selected_ds].index(condition["field"]) if condition["field"] in CSV_STRUCTURES[selected_ds] else 0,
                                key=f"gfield_{i}_{j}"
                            )
                        else:
                            st.text_input("Field", value=condition.get("field", ""), key=f"gfield_{i}_{j}")
                    with cols[2]:
                        st.selectbox(
                            "eligibilityPeriod",
                            ["N/A", "Rolling 30 days", "Rolling 60 days", "Rolling 90 days", "Current month"],
                            index=0 if condition.get("eligibilityPeriod") == "N/A" else 1,
                            key=f"gperiod_{i}_{j}"
                        )
                    with cols[3]:
                        st.selectbox(
                            "function",
                            ["N/A", "sum", "count", "avg", "max", "min"],
                            index=0 if condition.get("function") == "N/A" else 1,
                            key=f"gfunc_{i}_{j}"
                        )
                    with cols[4]:
                        st.selectbox(
                            "Operator",
                            ["=", ">", "<", ">=", "<=", "!=", "contains"],
                            index=0,
                            key=f"gop_{i}_{j}"
                        )
                    with cols[5]:
                        st.text_input(
                            "Value",
                            value=condition.get("value", ""),
                            key=f"gval_{i}_{j}"
                        )
                    
                    if j < len(rule_item.get("conditions", [])) - 1:
                        with cols[6]:
                            st.selectbox(
                                "Connector",
                                ["AND", "OR"],
                                index=0 if condition.get("connector", "AND") == "AND" else 1,
                                key=f"gconn_{i}_{j}"
                            )

def initialize_session_state():
    """Initialize all session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I can help you create mortgage holder rules. What criteria would you like to use?"}
        ]
    if "current_rule" not in st.session_state:
        st.session_state.current_rule = None
    if "confirmed" not in st.session_state:
        st.session_state.confirmed = False
    if "user_prompt" not in st.session_state:
        st.session_state.user_prompt = ""
    if "awaiting_confirmation" not in st.session_state:
        st.session_state.awaiting_confirmation = False
    if "awaiting_modification" not in st.session_state:
        st.session_state.awaiting_modification = False
    if "dataframes" not in st.session_state:
        st.session_state.dataframes = {}

def display_chat_message(role: str, content: str):
    """Display a chat message in the UI"""
    with st.chat_message(role):
        # Clean the content if it's user input
        if role == "user":
            content = clean_user_input(content)
        st.markdown(content)

def handle_user_confirmation(confirmation: bool):
    """Handle user confirmation or modification request"""
    if confirmation:
        st.session_state.confirmed = True
        st.session_state.awaiting_confirmation = False
        st.session_state.messages.append({"role": "assistant", "content": "Great! Here's your final rule:"})
    else:
        st.session_state.awaiting_confirmation = False
        st.session_state.awaiting_modification = True
        st.session_state.messages.append({"role": "assistant", "content": "What changes would you like to make to the rule?"})

def generate_new_rule():
    """Generate a new rule based on current state"""
    modification_request = None
    if st.session_state.awaiting_modification and st.session_state.messages[-1]["role"] == "user":
        modification_request = clean_user_input(st.session_state.messages[-1]["content"])
    
    with st.spinner("Generating rule..."):
        new_rule = generate_rule_with_llama(
            st.session_state.user_prompt,
            modification_request
        )
        
        if new_rule:
            st.session_state.current_rule = new_rule
            rule_preview = json.dumps(new_rule, indent=2)
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"I've generated this rule:\n\n```json\n{rule_preview}\n```\n\nDoes this meet your requirements?"
            })
            st.session_state.awaiting_confirmation = True
            st.session_state.awaiting_modification = False
        else:
            st.session_state.messages.append({
                "role": "assistant",
                "content": "I couldn't generate a valid rule. Could you please provide more details?"
            })

def main():
    st.set_page_config(page_title="Mortgage Rule Generator", layout="wide")
    st.title("🏦 Mortgage Rule Generator with Llama 3")
    
    # Custom CSS for better UI
    st.markdown("""
    <style>
        .stChatFloatingInputContainer {
            bottom: 20px;
        }
        .stChatMessage {
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 12px;
        }
        .assistant-message {
            background-color: #f0f2f6;
        }
        .user-message {
            background-color: #e3f2fd;
        }
        .stTextInput input, .stSelectbox select {
            font-size: 14px !important;
        }
        .stExpander {
            margin-bottom: 15px;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
        }
        .stExpander .streamlit-expanderHeader {
            font-weight: bold;
            background-color: #f5f5f5;
            padding: 10px 15px;
        }
        .stSelectbox > div > div {
            padding: 8px 12px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    initialize_session_state()
    
    # Create main layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Display rule UI
        if st.session_state.current_rule:
            display_rule_ui(st.session_state.current_rule)
            
            # Show final JSON if confirmed
            if st.session_state.confirmed:
                st.success("✅ Final Rule Confirmed")
                st.json(st.session_state.current_rule)
                
                # Add download button
                json_str = json.dumps(st.session_state.current_rule, indent=2)
                st.download_button(
                    label="Download Rule JSON",
                    data=json_str,
                    file_name=f"mortgage_rule_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
                
                if st.button("Create New Rule"):
                    # Reset for new rule
                    st.session_state.messages = [
                        {"role": "assistant", "content": "Let's create a new rule. What criteria would you like to use?"}
                    ]
                    st.session_state.current_rule = None
                    st.session_state.confirmed = False
                    st.session_state.user_prompt = ""
                    st.rerun()
    
    with col2:
        # Display chat messages
        st.subheader("Rule Assistant")
        
        for message in st.session_state.messages:
            display_chat_message(message["role"], message["content"])
        
        # Handle user input
        if prompt := st.chat_input("Type your message here..."):
            # Clean the user input first
            cleaned_prompt = clean_user_input(prompt)
            st.session_state.messages.append({"role": "user", "content": cleaned_prompt})
            display_chat_message("user", cleaned_prompt)
            
            # Determine what to do based on current state
            if not st.session_state.user_prompt:
                # First prompt - generate initial rule
                st.session_state.user_prompt = cleaned_prompt
                generate_new_rule()
                st.rerun()
            
            elif st.session_state.awaiting_confirmation:
                # User is responding to confirmation question
                if "yes" in cleaned_prompt.lower() or "correct" in cleaned_prompt.lower():
                    handle_user_confirmation(True)
                else:
                    handle_user_confirmation(False)
                st.rerun()
            
            elif st.session_state.awaiting_modification:
                # User is providing modification details
                generate_new_rule()
                st.rerun()
            
            else:
                # New conversation
                st.session_state.user_prompt = cleaned_prompt
                st.session_state.current_rule = None
                st.session_state.confirmed = False
                generate_new_rule()
                st.rerun()

if __name__ == "__main__":
    main()
