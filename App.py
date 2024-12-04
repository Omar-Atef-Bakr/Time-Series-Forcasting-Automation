import streamlit as st
from streamlit_chat import message
import google.generativeai as genai
import pandas as pd
import json
import io
from preprocess import preprocess_data, save_receipt  # Assuming preprocess.py is in the same directory

# Configure Google Generative AI
genai.configure(api_key="AIzaSyBZ2HnvtJ6ayRz0cztZTDdzOth_yu5ZiKA")
model = genai.GenerativeModel("gemini-1.5-flash")

# Initialize session state for the chatbot
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Function to handle chatbot interaction
def interact_with_chatbot(user_input):
    response = model.generate_content(user_input)
    return response.text

# Function to render the chatbot
def render_chatbot():
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 💬 Gemini Chatbot")
        st.markdown(
            "Ask me anything about this AutoML app or AI concepts! Expand to see full conversation."
        )
        expand_chat = st.checkbox("Expand Chat", value=False)

    if expand_chat:
        st.markdown("#### Chatbot Conversation")
        # Display the conversation history
        for i, message_pair in enumerate(st.session_state["messages"]):
            user_msg, bot_msg = message_pair
            message(user_msg, is_user=True, key=f"user_{i}")
            message(bot_msg, is_user=False, key=f"bot_{i}")

        # Input for the user to send a message
        user_input = st.text_input("Type your message:")
        if st.button("Send"):
            if user_input.strip():
                # Add user message to the session state
                st.session_state["messages"].append((user_input, ""))
                # Get chatbot response
                bot_response = interact_with_chatbot(user_input)
                # Update the session state with the bot's response
                st.session_state["messages"][-1] = (user_input, bot_response)

# State to track the current step
if "step" not in st.session_state:
    st.session_state["step"] = 1

# Function to render a step with a progress indicator
def render_step(step_num, description, completed):
    status_color = "green" if completed else "gray"
    step_indicator = f"""
    <div style='display: flex; align-items: center; margin-bottom: 10px;'>
        <div style='width: 30px; height: 30px; border-radius: 50%; background-color: {status_color}; color: white; display: flex; justify-content: center; align-items: center; font-weight: bold;'>
            {step_num}
        </div>
        <div style='margin-left: 10px; font-size: 16px; color: {status_color};'>
            {description}
        </div>
    </div>
    """
    st.markdown(step_indicator, unsafe_allow_html=True)

# Main page: Step-based workflow
def render_workflow_page():
    st.title("AutoML Workflow")
    st.markdown("Follow the steps to complete your AutoML pipeline.")

    # Step 1: Upload dataset
    render_step(1, "Upload your dataset", st.session_state["step"] > 1)
    if st.session_state["step"] == 1:
        uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])
        if uploaded_file:
            st.success("Dataset uploaded successfully!")
            st.session_state["step"] = 2
            # Store the uploaded file in session state for later use
            st.session_state["uploaded_file"] = uploaded_file

    # Step 2: Preprocessing
    render_step(2, "Preprocessing your data", st.session_state["step"] > 2)
    if st.session_state["step"] == 2:
        if "uploaded_file" in st.session_state:
            with st.spinner("Preprocessing your dataset... This may take a while."):
                # Preprocess the uploaded file
                uploaded_file = st.session_state["uploaded_file"]
                file_path = "temp_dataset.csv"  # Temporarily saving it for preprocessing
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Preprocess the data using preprocess.py logic
                df, receipt = preprocess_data(file_path)

                # Save the receipt
                save_receipt(receipt, "preprocessing_receipt.json")

                # Send the receipt to Gemini for a detailed report
                report_prompt = f"Here is the preprocessing receipt in JSON format: {json.dumps(receipt)}. use this json file to give keypoints about what happend , be precise , short and."
                report = interact_with_chatbot(report_prompt)

                # Display the receipt and report
                st.subheader("Preprocessing Receipt")
                st.json(receipt)
                st.subheader("Preprocessing Report")
                st.markdown(report)

                # Update the workflow step
                st.session_state["step"] = 3
        else:
            st.warning("Please upload a dataset first!")

    # Step 3: Model Training
    render_step(3, "Train your model", st.session_state["step"] > 3)
    if st.session_state["step"] == 3:
        st.write("Train your model with the prepared data.")
        if st.button("Finish Training"):
            st.success("Model training completed!")
            st.session_state["step"] = 4

    # Step 4: Model Evaluation
    render_step(4, "Evaluate your model", st.session_state["step"] > 4)
    if st.session_state["step"] == 4:
        st.write("Evaluate your model performance.")
        if st.button("Finish Evaluation"):
            st.success("Model evaluation completed!")
            st.session_state["step"] = 5

    # Step 5: Deployment
    render_step(5, "Deploy your model", st.session_state["step"] > 5)
    if st.session_state["step"] == 5:
        st.write("Deploy your model as an API or web service.")
        if st.button("Finish Deployment"):
            st.success("Deployment completed!")
            st.session_state["step"] = 6

    if st.session_state["step"] > 5:
        st.success("🎉 All steps completed!")

# Other pages
def render_your_work_page():
    st.title("Your Work")
    st.write("Here, you can track your progress and manage your tasks.")

def render_your_datasets_page():
    st.title("Your Datasets")
    st.write("Manage and view your datasets here.")

def render_your_models_page():
    st.title("Your Models")
    st.write("Manage your saved models and their versions.")

def render_settings_page():
    st.title("Settings")
    st.write("Configure application settings.")

# Main function
def main():
    st.set_page_config(page_title="AutoML Website", layout="wide")

    # Sidebar navigation
    st.sidebar.title("Navigation")
    tabs = st.sidebar.radio(
        "Go to",
        ["Workflow", "Your Work", "Your Datasets", "Your Models", "Settings"]
    )

    # Render pages based on the selected tab
    if tabs == "Workflow":
        render_workflow_page()
    elif tabs == "Your Work":
        render_your_work_page()
    elif tabs == "Your Datasets":
        render_your_datasets_page()
    elif tabs == "Your Models":
        render_your_models_page()
    elif tabs == "Settings":
        render_settings_page()

    # Always render the chatbot
    render_chatbot()

# Entry point
if __name__ == "__main__":
    main()
