import streamlit as st
import pandas as pd
import json
import re
import os
import openai
from openai import OpenAI

# Initialize session state
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "system", "content": "You are a financial rule builder assistant."},
            {"role": "assistant", "content": "Describe the rule you want to create in natural language."}
        ]
    if "rule_json" not in st.session_state:
        st.session_state.rule_json = None
    if "confirmed" not in st.session_state:
        st.session_state.confirmed = False
    if "schema" not in st.session_state:
        st.session_state.schema = load_schema()

# Load CSV schemas
def load_schema():
    schema = {}
    data_dir = "data"
    for file in os.listdir(data_dir):
        if file.endswith(".csv"):
            try:
                df = pd.read_csv(os.path.join(data_dir, file))
                schema[file.replace("sample_", "").replace(".csv", "")] = list(df.columns)
            except Exception as e:
                st.error(f"Error loading {file}: {str(e)}")
    return schema

# Generate rule using OpenAI
def generate_rule(prompt, schema):
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    
    system_prompt = f"""
    You are an expert financial rule builder. Create JSON rules based on user descriptions using these CSV schemas:
    {json.dumps(schema, indent=2)}
    
    Rule Structure:
    - Top-level list of rule objects
    - Each rule object can be either a 'condition' or 'conditionGroup'
    - Conditions have: dataSource, field, eligibilityPeriod, function, operator, value, connector (AND/OR for non-last items)
    - Condition groups contain lists of conditions
    
    Example:
    {{
      "rules": [
        {{
          "id": "condition1",
          "dataSource": "savings_account_transactions",
          "field": "amount",
          "eligibilityPeriod": "Rolling 30 days",
          "function": "sum",
          "operator": ">",
          "value": "2500",
          "ruleType": "condition",
          "connector": "AND"
        }},
        {{
          "id": "group1",
          "ruleType": "conditionGroup",
          "conditions": [
            {{
              "id": "condition2",
              "dataSource": "mortgage_accounts",
              "field": "status",
              "eligibilityPeriod": "N/A",
              "function": "N/A",
              "operator": "=",
              "value": "active",
              "connector": "OR"
            }},
            {{
              "id": "condition3",
              "dataSource": "loan_repayments",
              "field": "balance",
              "eligibilityPeriod": "N/A",
              "function": "N/A",
              "operator": ">",
              "value": "1000"
            }}
          ]
        }}
      ]
    }}
    
    Output ONLY valid JSON. Do not include any explanations.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        st.error(f"Error generating rule: {str(e)}")
        return None

# Display rule in UI
def display_rule(rule_data):
    if not rule_data or "rules" not in rule_data:
        st.warning("No valid rule structure found")
        return
    
    with st.expander("Generated Rule", expanded=True):
        for i, rule in enumerate(rule_data["rules"]):
            if rule["ruleType"] == "condition":
                st.markdown(f"""
                **Condition {i+1}**
                - Data Source: `{rule['dataSource']}`
                - Field: `{rule['field']}`
                - Function: `{rule['function']}`
                - Operator: `{rule['operator']}`
                - Value: `{rule['value']}`
                - Period: `{rule['eligibilityPeriod']}`
                {f"- Connector: `{rule['connector']}`" if 'connector' in rule else ""}
                """)
            elif rule["ruleType"] == "conditionGroup":
                with st.expander(f"Condition Group {i+1}"):
                    for j, condition in enumerate(rule["conditions"]):
                        st.markdown(f"""
                        **Condition {j+1}**
                        - Data Source: `{condition['dataSource']}`
                        - Field: `{condition['field']}`
                        - Function: `{condition['function']}`
                        - Operator: `{condition['operator']}`
                        - Value: `{condition['value']}`
                        - Period: `{condition['eligibilityPeriod']}`
                        {f"- Connector: `{condition['connector']}`" if 'connector' in condition else ""}
                        """)
        st.divider()
        st.json(rule_data)

# Main app
def main():
    st.set_page_config(layout="wide", page_title="Financial Rule Builder")
    st.title("Financial Rule Builder")
    init_session_state()
    
    # Create columns
    left_col, right_col = st.columns([2, 1])
    
    # Left column - Rule display
    with left_col:
        st.header("Rule Configuration")
        if st.session_state.rule_json:
            display_rule(st.session_state.rule_json)
        else:
            st.info("No rule generated yet. Describe your rule in the chat.")
    
    # Right column - Chat interface
    with right_col:
        st.header("Rule Builder Assistant")
        
        # Display chat messages
        for msg in st.session_state.messages:
            if msg["role"] != "system":
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])
        
        # Chat input
        if prompt := st.chat_input("Describe your rule (e.g., Customer spends over $2500...)"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)
            
            # Generate rule
            with st.spinner("Building your rule..."):
                rule_json = generate_rule(prompt, st.session_state.schema)
            
            if rule_json:
                st.session_state.rule_json = rule_json
                response = "Here's your rule. Does this match your requirements?"
                st.session_state.messages.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.write(response)
                    st.button("Confirm Rule", key="confirm_btn")
            else:
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": "Sorry, I couldn't generate a rule. Please try again with more details."
                })
        
        # Confirmation and editing
        if st.session_state.rule_json:
            if st.button("Confirm Rule", key="confirm_btn_main"):
                st.session_state.confirmed = True
                st.success("Rule confirmed! Ready for production use.")
                st.balloons()
            
            if st.button("Edit Rule", key="edit_btn"):
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": "What changes would you like to make?"
                })
                st.session_state.confirmed = False
                st.experimental_rerun()

if __name__ == "__main__":
    main()
