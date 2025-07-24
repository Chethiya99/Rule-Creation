import streamlit as st
import pandas as pd
import groq
import json
from typing import List, Dict, Any, Optional
import time
from datetime import datetime

# Initialize Groq client
try:
    client = groq.Client(api_key=st.secrets["GROQ_API_KEY"])
except Exception as e:
    st.error(f"Failed to initialize Groq client: {str(e)}")
    st.stop()

# Define CSV file structures with exact column names
CSV_STRUCTURES = {
    "sample_mortgage_accounts.csv": ["customer_id", "account_status", "loan_amount", "interest_rate", "start_date", "term"],
    "sample_loan_repayments.csv": ["transaction_id", "customer_id", "payment_amount", "payment_date", "loan_id", "status"],
    "sample_telco_billing.csv": ["bill_id", "customer_id", "amount", "due_date", "paid_date", "service_type"],
    "sample_product_enrollments.csv": ["enrollment_id", "customer_id", "product_id", "enrollment_date", "status"],
    "sample_customer_profiles.csv": ["customer_id", "name", "age", "income", "credit_score", "address"],
    "sample_savings_account_transactions.csv": ["transaction_id", "account_id", "customer_id", "amount", "date", "transaction_type"],
    "sample_credit_card_transactions.csv": ["transaction_id", "card_id", "customer_id", "amount", "date", "merchant", "category"]
}

# Create mapping for case-insensitive matching
COLUMN_MAPPING = {}
for file, columns in CSV_STRUCTURES.items():
    for col in columns:
        COLUMN_MAPPING[col.lower()] = col

def clean_user_input(text: str) -> str:
    """Clean user input by removing extra spaces between characters"""
    text = ' '.join(text.split())
    return text.replace(" o n ", " on ").replace(" A N D ", " AND ").replace(" O R ", " OR ")

def generate_prompt_guidance(user_input: str, modification_request: Optional[str] = None) -> str:
    """Generate guidance for the AI with emphasis on exact column names"""
    available_data = "\n".join([f"- {f}: {', '.join(cols)}" for f, cols in CSV_STRUCTURES.items()])
    
    base_prompt = f"""
    You are a financial rule generation assistant. Your task is to help create rules for mortgage holders based on available data sources.

    CRITICAL INSTRUCTIONS:
    1. You MUST use ONLY the exact column names from the available data sources
    2. Field names are case-sensitive and must match exactly as provided
    3. If a similar concept exists but with different naming, use the provided column name
    4. For rules with multiple conditions, use a SINGLE conditionGroup with connector "AND" or "OR"
    5. DO NOT create top-level conditions when a conditionGroup is needed

    Available data sources and their EXACT columns:
    {available_data}

    The user has provided this requirement: "{user_input}"
    """
    
    if modification_request:
        base_prompt += f"\nThe user requested these modifications: {modification_request}"
    
    base_prompt += """
    Analyze this requirement and:
    1. Identify which data sources are needed
    2. Use ONLY the exact column names from the sources
    3. Create a logical rule structure with proper AND/OR conditions
    4. Include all these fields for each condition:
       - dataSource (file name exactly as shown)
       - field (column name exactly as shown)
       - eligibilityPeriod (use "Rolling 30 days" for time-based conditions, otherwise "N/A")
       - function (use "sum", "count", "avg" where appropriate, otherwise "N/A")
       - operator
       - value
    5. Output the rule in JSON format matching this schema:
        {
            "rules": [
                {
                    "id": "generated_id",
                    "dataSource": "source_name",
                    "field": "column_name",
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

    IMPORTANT: For rules with multiple conditions:
    - Use a SINGLE conditionGroup that contains all the conditions
    - Set connector for the conditionGroup (AND/OR)
    - DO NOT create separate top-level conditions

    Respond ONLY with the JSON output. Do not include any additional explanation or markdown formatting.
    The rule should be as specific as possible to match the user's requirements.
    """
    
    return base_prompt

def validate_and_correct_columns(rule: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure all field names match the exact column names from CSV structures"""
    if not rule or "rules" not in rule:
        return rule
    
    # First pass: correct top-level rules
    for rule_item in rule["rules"]:
        # Handle condition groups
        if rule_item.get("ruleType") == "conditionGroup":
            for condition in rule_item.get("conditions", []):
                condition["field"] = correct_column_name(condition.get("field", ""))
        # Handle individual conditions
        else:
            rule_item["field"] = correct_column_name(rule_item.get("field", ""))
    
    # Second pass: remove redundant top-level conditions when conditionGroup exists
    rules = rule["rules"]
    new_rules = []
    condition_group_exists = any(r.get("ruleType") == "conditionGroup" for r in rules)
    
    if condition_group_exists:
        # Only keep conditionGroups and remove standalone conditions
        new_rules = [r for r in rules if r.get("ruleType") == "conditionGroup"]
    else:
        new_rules = rules
    
    rule["rules"] = new_rules
    return rule

def correct_column_name(field_name: str) -> str:
    """Convert field name to exact match from CSV columns"""
    if not field_name:
        return field_name
    
    # Try exact match first
    if field_name in COLUMN_MAPPING.values():
        return field_name
    
    # Try case-insensitive match
    lower_name = field_name.lower()
    if lower_name in COLUMN_MAPPING:
        return COLUMN_MAPPING[lower_name]
    
    # Try common variations
    variations = {
        "id": "customer_id",
        "payment": "payment_amount",
        "date": "payment_date",
        "status": "account_status",
        "amount": "payment_amount",
        "score": "credit_score",
        "type": "transaction_type",
        "transaction": "amount",
        "spend": "amount",
        "minutes": "service_type"  # For telco example
    }
    
    for partial, full in variations.items():
        if partial in lower_name:
            return full
    
    return field_name  # Return original if no match found

def generate_rule_with_llama(user_input: str, modification_request: Optional[str] = None) -> Dict[str, Any]:
    """Use Groq/Llama to generate a rule based on user input"""
    prompt = generate_prompt_guidance(user_input, modification_request)
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a financial rule generation expert that creates precise JSON rules using EXACT column names from provided data sources. For multiple conditions, use a SINGLE conditionGroup."
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
        json_start = response_content.find('{')
        json_end = response_content.rfind('}') + 1
        if json_start == -1 or json_end == 0:
            st.error("Invalid JSON response from model")
            return None
            
        json_str = response_content[json_start:json_end]
        rule = json.loads(json_str)
        
        # Validate and correct column names
        return validate_and_correct_columns(rule)
    
    except Exception as e:
        st.error(f"Error generating rule: {str(e)}")
        return None

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
                    # Data source dropdown with exact file names
                    selected_ds = st.selectbox(
                        "Data Source",
                        options=list(CSV_STRUCTURES.keys()),
                        index=list(CSV_STRUCTURES.keys()).index(rule_item["dataSource"]) 
                        if rule_item["dataSource"] in CSV_STRUCTURES else 0,
                        key=f"ds_{i}"
                    )
                with cols[1]:
                    # Field dropdown with exact column names for selected data source
                    columns = CSV_STRUCTURES.get(selected_ds, [])
                    selected_field = st.selectbox(
                        "Field", 
                        options=columns,
                        index=columns.index(rule_item["field"]) 
                        if rule_item["field"] in columns else 0,
                        key=f"field_{i}"
                    )
                with cols[2]:
                    period_index = ["N/A", "Rolling 30 days", "Rolling 60 days", "Rolling 90 days", "Current month"].index(
                        rule_item.get("eligibilityPeriod", "N/A")
                    ) if rule_item.get("eligibilityPeriod", "N/A") in ["N/A", "Rolling 30 days", "Rolling 60 days", "Rolling 90 days", "Current month"] else 0
                    st.selectbox("eligibilityPeriod", 
                                ["N/A", "Rolling 30 days", "Rolling 60 days", "Rolling 90 days", "Current month"],
                                index=period_index,
                                key=f"period_{i}")
                with cols[3]:
                    func_index = ["N/A", "sum", "count", "avg", "max", "min"].index(
                        rule_item.get("function", "N/A")
                    ) if rule_item.get("function", "N/A") in ["N/A", "sum", "count", "avg", "max", "min"] else 0
                    st.selectbox("function", 
                                ["N/A", "sum", "count", "avg", "max", "min"],
                                index=func_index,
                                key=f"func_{i}")
                with cols[4]:
                    op_options = ["=", ">", "<", ">=", "<=", "!=", "contains"]
                    op_index = op_options.index(rule_item.get("operator", "=")) if rule_item.get("operator", "=") in op_options else 0
                    st.selectbox("Operator", 
                                op_options,
                                index=op_index,
                                key=f"op_{i}")
                with cols[5]:
                    st.text_input("Value", value=str(rule_item.get("value", "")), 
                                key=f"val_{i}")
                
                if i < len(rule["rules"]) - 1:
                    with cols[6]:
                        connector_options = ["AND", "OR"]
                        conn_index = connector_options.index(rule_item.get("connector", "AND")) if rule_item.get("connector", "AND") in connector_options else 0
                        st.selectbox("Connector", 
                                    connector_options,
                                    index=conn_index,
                                    key=f"conn_{i}")
        
        elif rule_item.get("ruleType") == "conditionGroup":
            with st.expander(f"Condition Group", expanded=True):
                # Display group-level connector
                group_connector = rule_item.get("connector", "AND")
                st.selectbox("Group Connector", 
                            ["AND", "OR"],
                            index=0 if group_connector == "AND" else 1,
                            key=f"group_conn_{i}")
                
                # Display conditions within the group
                for j, condition in enumerate(rule_item.get("conditions", [])):
                    st.markdown(f"**Condition {j+1}**")
                    cols = st.columns(7)
                    with cols[0]:
                        selected_ds = st.selectbox(
                            "Data Source",
                            options=list(CSV_STRUCTURES.keys()),
                            index=list(CSV_STRUCTURES.keys()).index(condition["dataSource"]) 
                            if condition["dataSource"] in CSV_STRUCTURES else 0,
                            key=f"gds_{i}_{j}"
                        )
                    with cols[1]:
                        columns = CSV_STRUCTURES.get(selected_ds, [])
                        selected_field = st.selectbox(
                            "Field", 
                            options=columns,
                            index=columns.index(condition["field"]) 
                            if condition["field"] in columns else 0,
                            key=f"gfield_{i}_{j}"
                        )
                    with cols[2]:
                        period_index = ["N/A", "Rolling 30 days", "Rolling 60 days", "Rolling 90 days", "Current month"].index(
                            condition.get("eligibilityPeriod", "N/A")
                        ) if condition.get("eligibilityPeriod", "N/A") in ["N/A", "Rolling 30 days", "Rolling 60 days", "Rolling 90 days", "Current month"] else 0
                        st.selectbox("eligibilityPeriod", 
                                    ["N/A", "Rolling 30 days", "Rolling 60 days", "Rolling 90 days", "Current month"],
                                    index=period_index,
                                    key=f"gperiod_{i}_{j}")
                    with cols[3]:
                        func_index = ["N/A", "sum", "count", "avg", "max", "min"].index(
                            condition.get("function", "N/A")
                        ) if condition.get("function", "N/A") in ["N/A", "sum", "count", "avg", "max", "min"] else 0
                        st.selectbox("function", 
                                    ["N/A", "sum", "count", "avg", "max", "min"],
                                    index=func_index,
                                    key=f"gfunc_{i}_{j}")
                    with cols[4]:
                        op_options = ["=", ">", "<", ">=", "<=", "!=", "contains"]
                        op_index = op_options.index(condition.get("operator", "=")) if condition.get("operator", "=") in op_options else 0
                        st.selectbox("Operator", 
                                    op_options,
                                    index=op_index,
                                    key=f"gop_{i}_{j}")
                    with cols[5]:
                        st.text_input("Value", value=str(condition.get("value", "")), 
                                    key=f"gval_{i}_{j}")
                    
                    if j < len(rule_item.get("conditions", [])) - 1:
                        with cols[6]:
                            connector_options = ["AND", "OR"]
                            conn_index = connector_options.index(condition.get("connector", "AND")) if condition.get("connector", "AND") in connector_options else 0
                            st.selectbox("Connector", 
                                        connector_options,
                                        index=conn_index,
                                        key=f"gconn_{i}_{j}")

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

def display_chat_message(role: str, content: str):
    """Display a chat message in the UI"""
    with st.chat_message(role):
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
        .condition-group {
            border-left: 3px solid #4285f4;
            padding-left: 15px;
            margin-top: 10px;
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
