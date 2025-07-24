import streamlit as st
import pandas as pd
import groq
import json
from typing import List, Dict, Any, Optional
from datetime import datetime

# Initialize Groq client
try:
    client = groq.Client(api_key=st.secrets["GROQ_API_KEY"])
except Exception as e:
    st.error(f"Failed to initialize Groq client: {str(e)}")
    st.stop()

# Define CSV file structures
CSV_STRUCTURES = {
    "sample_mortgage_accounts.csv": ["customer_id", "account_status", "loan_amount", "interest_rate", "start_date", "term"],
    "sample_loan_repayments.csv": ["transaction_id", "customer_id", "payment_amount", "payment_date", "loan_id", "status"],
    "sample_telco_billing.csv": ["bill_id", "customer_id", "amount", "due_date", "paid_date", "service_type"],
    "sample_product_enrollments.csv": ["enrollment_id", "customer_id", "product_id", "enrollment_date", "status"],
    "sample_customer_profiles.csv": ["customer_id", "name", "age", "income", "credit_score", "address"],
    "sample_savings_account_transactions.csv": ["transaction_id", "account_id", "customer_id", "amount", "date", "transaction_type"],
    "sample_credit_card_transactions.csv": ["transaction_id", "card_id", "customer_id", "amount", "date", "merchant", "category"]
}

def clean_user_input(text: str) -> str:
    text = ' '.join(text.split())
    return text.replace(" o n ", " on ").replace(" A N D ", " AND ").replace(" O R ", " OR ")

def generate_prompt_guidance(user_input: str, modification_request: Optional[str] = None) -> str:
    available_data = "\n".join([
        f"- {file}: {', '.join(cols)}" for file, cols in CSV_STRUCTURES.items()
    ])
    
    base_prompt = f"""
You are a financial rule generation assistant.

Your task is to help create precise logical rules for mortgage holders based on available data sources.

IMPORTANT:
- ONLY use the **exact** column names provided in the list of CSV data sources below.
- Do NOT invent or infer new column names.
- Match column names to the correct data source.
- Ensure proper use of AND/OR logic if multiple conditions exist.

Available data sources and columns:
{available_data}

The user has provided the following requirement:
"{user_input}"
"""
    if modification_request:
        base_prompt += f"\nUser requested the following modifications: {modification_request}"

    base_prompt += """
Now analyze this requirement and:
1. Identify which data sources are needed
2. Choose ONLY valid column names from the corresponding source
3. Construct a rule using this format:
   {
       "rules": [
           {
               "id": "generated_id",
               "dataSource": "source_name",
               "field": "column_name_from_source",
               "eligibilityPeriod": "Rolling 30 days" or "N/A",
               "function": "sum", "count", "avg" or "N/A",
               "operator": ">", "<", ">=", "<=", "=", "!=", "contains",
               "value": "comparison_value",
               "priority": null,
               "ruleType": "condition" or "conditionGroup",
               "connector": "AND"/"OR"/null,
               "conditions": [ /* for conditionGroup only */ ]
           }
       ]
   }

⚠️ Strict Rule: All `field` values **must match exactly** the column names from the CSV structures above.

Respond ONLY with valid JSON output. No explanation, no markdown. Just JSON.
"""
    return base_prompt

def generate_rule_with_llama(user_input: str, modification_request: Optional[str] = None) -> Dict[str, Any]:
    prompt = generate_prompt_guidance(user_input, modification_request)
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a financial rule generation expert that creates precise JSON rules based on data sources."},
                {"role": "user", "content": prompt}
            ],
            model="llama3-70b-8192",
            temperature=0.3
        )
        response_content = chat_completion.choices[0].message.content
        json_str = response_content[response_content.find('{'):response_content.rfind('}')+1]
        return json.loads(json_str)
    except Exception as e:
        st.error(f"Error generating rule: {str(e)}")
        return None

def display_rule_ui(rule: Dict[str, Any]) -> None:
    if not rule or "rules" not in rule:
        st.warning("No valid rule generated yet")
        return
    
    st.subheader("Rule Conditions")
    st.markdown("Define the logical conditions for this rule to apply.")
    st.checkbox("Enable priority order and drag & drop", value=False, key="priority_order")
    
    for i, rule_item in enumerate(rule["rules"]):
        if rule_item.get("ruleType") == "condition":
            with st.expander(f"Condition {i+1}", expanded=True):
                cols = st.columns(7)
                with cols[0]: st.text_input("Data Source", value=rule_item.get("dataSource", ""), key=f"ds_{i}")
                with cols[1]: st.text_input("Field", value=rule_item.get("field", ""), key=f"field_{i}")
                with cols[2]: st.selectbox("eligibilityPeriod", ["N/A", "Rolling 30 days", "Rolling 60 days", "Rolling 90 days", "Current month"], index=0 if rule_item.get("eligibilityPeriod") == "N/A" else 1, key=f"period_{i}")
                with cols[3]: st.selectbox("function", ["N/A", "sum", "count", "avg", "max", "min"], index=0 if rule_item.get("function") == "N/A" else 1, key=f"func_{i}")
                with cols[4]: st.selectbox("Operator", ["=", ">", "<", ">=", "<=", "!=", "contains"], index=0, key=f"op_{i}")
                with cols[5]: st.text_input("Value", value=rule_item.get("value", ""), key=f"val_{i}")
                if i < len(rule["rules"]) - 1:
                    with cols[6]: st.selectbox("Connector", ["AND", "OR"], index=0 if rule_item.get("connector", "AND") == "AND" else 1, key=f"conn_{i}")
        
        elif rule_item.get("ruleType") == "conditionGroup":
            with st.expander(f"Condition Group {i+1}", expanded=True):
                st.markdown("#### Condition Group")
                for j, condition in enumerate(rule_item.get("conditions", [])):
                    cols = st.columns(7)
                    with cols[0]: st.text_input("Data Source", value=condition.get("dataSource", ""), key=f"gds_{i}_{j}")
                    with cols[1]: st.text_input("Field", value=condition.get("field", ""), key=f"gfield_{i}_{j}")
                    with cols[2]: st.selectbox("eligibilityPeriod", ["N/A", "Rolling 30 days", "Rolling 60 days", "Rolling 90 days", "Current month"], index=0 if condition.get("eligibilityPeriod") == "N/A" else 1, key=f"gperiod_{i}_{j}")
                    with cols[3]: st.selectbox("function", ["N/A", "sum", "count", "avg", "max", "min"], index=0 if condition.get("function") == "N/A" else 1, key=f"gfunc_{i}_{j}")
                    with cols[4]: st.selectbox("Operator", ["=", ">", "<", ">=", "<=", "!=", "contains"], index=0, key=f"gop_{i}_{j}")
                    with cols[5]: st.text_input("Value", value=condition.get("value", ""), key=f"gval_{i}_{j}")
                    if j < len(rule_item.get("conditions", [])) - 1:
                        with cols[6]: st.selectbox("Connector", ["AND", "OR"], index=0 if condition.get("connector", "AND") == "AND" else 1, key=f"gconn_{i}_{j}")

def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I can help you create mortgage holder rules. What criteria would you like to use?"}]
    if "current_rule" not in st.session_state: st.session_state.current_rule = None
    if "confirmed" not in st.session_state: st.session_state.confirmed = False
    if "user_prompt" not in st.session_state: st.session_state.user_prompt = ""
    if "awaiting_confirmation" not in st.session_state: st.session_state.awaiting_confirmation = False
    if "awaiting_modification" not in st.session_state: st.session_state.awaiting_modification = False

def display_chat_message(role: str, content: str):
    with st.chat_message(role):
        if role == "user": content = clean_user_input(content)
        st.markdown(content)

def handle_user_confirmation(confirmation: bool):
    if confirmation:
        st.session_state.confirmed = True
        st.session_state.awaiting_confirmation = False
        st.session_state.messages.append({"role": "assistant", "content": "Great! Here's your final rule:"})
    else:
        st.session_state.awaiting_confirmation = False
        st.session_state.awaiting_modification = True
        st.session_state.messages.append({"role": "assistant", "content": "What changes would you like to make to the rule?"})

def generate_new_rule():
    modification_request = None
    if st.session_state.awaiting_modification and st.session_state.messages[-1]["role"] == "user":
        modification_request = clean_user_input(st.session_state.messages[-1]["content"])
    
    with st.spinner("Generating rule..."):
        new_rule = generate_rule_with_llama(st.session_state.user_prompt, modification_request)
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
            st.session_state.messages.append({"role": "assistant", "content": "I couldn't generate a valid rule. Could you please provide more details?"})

def main():
    st.set_page_config(page_title="Mortgage Rule Generator", layout="wide")
    st.title("🏦 Mortgage Rule Generator with Llama 3")

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

    initialize_session_state()
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
                    st.session_state.messages = [{"role": "assistant", "content": "Let's create a new rule. What criteria would you like to use?"}]
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
