import pandas as pd
import torch
import io
import os
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# --- 1. Global Setup (Done ONCE on startup) ---
# ... (This is all your existing setup code) ...

print("Setting up cache directories...")
os.environ["HUGGINGFACE_HUB_CACHE"] = "models"
os.environ["TORCH_HOME"] = "models"

# --- Load Data & Get Schema ---
print("Loading DataFrame and schema...")
try:
    df = pd.read_csv('data/employee.csv')
    buffer = io.StringIO()
    df.info(buf=buffer)
    df_structure = buffer.getvalue()
    print("--- DataFrame Schema Loaded ---")
    print(df_structure)
except FileNotFoundError:
    print("FATAL ERROR: 'data/employee.csv' not found.")
    print("Please make sure the 'data' folder is in the same directory as app.py")
    exit()
except Exception as e:
    print(f"FATAL ERROR: Could not load data: {e}")
    exit()

# --- Load Model & Pipeline ---
print("Loading Phi-2 model. This may take a few minutes...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_name = "microsoft/phi-2"

try:
    model_llm = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    llm_pipeline = pipeline(
        "text-generation",
        model=model_llm,
        tokenizer=tokenizer,
        device=device,
        max_length=500
    )
    print("✅ Model and pipeline loaded successfully.")
except Exception as e:
    print(f"FATAL ERROR: Could not load model: {e}")
    print("Check internet connection or model name.")
    exit()


# --- get_answer Function (From your original code) ---
# --- get_answer Function (From your original code) ---
def get_answer(user_query):
    """
    Runs the exact two-role logic from the notebook.
    Returns:
        tuple: (friendly_answer, data_result_object)
    """

    # --- [Role 1] Text-to-Code ---
    prompt_role_1 = f"""
You are an expert Python assistant. You translate natural language questions into a
single, executable line of Python code for a pandas DataFrame named `df`.
Do not include any explanation, just the code.

DataFrame Schema:
{df_structure}

Examples:

Question: "How many employees are there?"
Code: len(df)

Question: "Show me the top 3 highest paid employees."
Code: df.sort_values('empSalary', ascending=False).head(3)

Question: "What is the total salary of the 'Sales' department?"
Code: df[df['empDept'] == 'Sales']['empSalary'].sum()

Question: "Give me all employees in 'Engineering'."
Code: df[df['empDept'] == 'Engineering']

Current Task:

Question: "{user_query}"
Code:"""

    print("--- [Role 1] Sending Text-to-Code Prompt ---")
    generated_code = ""
    data_result = None

    try:
        sequences = llm_pipeline(
            prompt_role_1,
            do_sample=False,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=100,
            return_full_text=False
        )
        response_only = sequences[0]['generated_text'].replace(prompt_role_1, "").strip()
        generated_code = response_only.split('\n')[0].strip()
        print(f"   > Extracted Code: {generated_code}")

        if "import" in generated_code or "os." in generated_code:
            raise PermissionError("Generated code contains unsafe commands.")

        data_result = eval(generated_code)
        print(f"   > Captured Data: {data_result}")

    except Exception as e:
        print(f"--- [Role 1] Error: {e} ---")
        # This is our consistent 2-item return for an error
        return f"Error in code generation: {e}", "No data"

    # --- [Role 2] Data-to-Text ---

    # This block now handles all data types, including your 'if/else' logic
    if data_result is not None:

        # --- [NEW] This is the if/else logic you asked for ---
        instruction_text = ""

        if isinstance(data_result, (pd.DataFrame, pd.Series)):
            # If it's a DataFrame or Series, give this instruction
            instruction_text = """Question: "Show me the top 3 highest paid employees.
        Data Result:
             empID   empName  empDept  empSalary
        10  E11    Karan    Sales      65000
        1    E02    Jane     Dev        60000
        15  E16    Ravi     Dev        60000
        Answer: Here are the top 3 highest paid employees."""
        else:
            # If it's a number, string, or anything else, give this instruction
            instruction_text = """
            Question: "What is the total salary of the 'Sales' department?"
            Data Result: 95000
            Answer: The total salary for the Sales department is 95,000.
            """
        # --- [END NEW] ---

        # We build the prompt with your new instruction *and* the examples
        prompt_role_2 = f"""
        From the example below reply the answer of the current task.
        {instruction_text}

        Current Task:

        Question: "{user_query}"
        Data Result: {data_result}
        Answer:"""

        print("\n--- [Role 2] Sending Data-to-Text Prompt ---")
        try:
            sequences_2 = llm_pipeline(
                prompt_role_2,
                do_sample=False,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens=200,
                return_full_text=False
            )
            friendly_answer = sequences_2[0]['generated_text'].strip().split('\n')[0].strip()
            print(f"\n--- Final Answer: {friendly_answer} ---")

            # Consistent 2-item return on success
            return friendly_answer, data_result

        except Exception as e:
            print(f"--- [Role 2] Error: {e} ---")
            # Consistent 2-item return on error
            return f"Error in answer summarization: {e}", data_result

    # This is the default "couldn't process it" if data_result is None
    # Consistent 2-item return
    return "Could not determine an answer.", "No data"


# --- chat_respond Function (From our previous conversation) ---
def chat_respond(message, chat_history):
    """
    This function is called by the gr.Blocks UI.
    It appends all outputs (answer, code, data)
    directly into the chat history as separate bubbles.
    """

    # 1. Call your main logic function
    friendly_answer, data_result = get_answer(message)

    # 2. Append the user message and the friendly answer
    chat_history.append((message, friendly_answer))

    # 3. Create the collapsible "Under the Hood" details
    # code_md = f"```python\n{code}\n```"
    data_md = ""

    if isinstance(data_result, pd.DataFrame):
        if data_result.empty:
            data_md = "\n`Empty DataFrame`"
        else:
            data_md = f"\n{data_result.to_markdown(index=False)}"
    # else:
    #     data_md = f"\n`{str(data_result)}`"

    details_md = f"""
    {data_md}
    """

    # 4. Append the new "details" bubble to the chat
    chat_history.append((None, details_md))

    # 5. Return the updated history and clear the user's input box
    return chat_history, ""


# --- [NEW] Improved ChatGPT-style gr.Blocks UI ---
print("Launching Gradio UI...")
with gr.Blocks(
        theme="soft",
        css="""
        /* Make the whole app fill the viewport height */
        .gradio-container { min-height: 100vh !important; }

        /* Make the chatbot window tall */
        #chatbot {
            height: 85vh;
            overflow-y: auto; /* Ensure it scrolls */

            /* --- 1. NEW MESSAGE BUBBLE COLORS (via variables) --- */
            --bubble-radius: 20px;
            --user-bubble-background: #3B82F6; /* Blue-600 */
            --user-bubble-text-color: white;
            --bot-bubble-background: #E5E7EB;  /* Gray-200 */
            --bot-bubble-text-color: #1F2937;  /* Gray-800 */
            /* --- End Bubble Colors --- */
        }

        /* --- 2. NEW iMessage/WhatsApp-style alignment --- */

        /* This targets the container for EACH message row */
        #chatbot .message-wrap {
            width: 100%;
            display: flex; /* Use flexbox for alignment */
        }

        /* This targets the USER message row */
        #chatbot .message-wrap.user {
            justify-content: flex-end; /* Pushes the bubble to the right */
        }

        /* This targets the BOT message row */
        #chatbot .message-wrap.bot {
            justify-content: flex-start; /* Pushes the bubble to the left (default) */
        }

        /* This makes the bubbles fit their content, not span the full width */
        #chatbot .message {
            width: auto;
            max-width: 80%; /* Don't let bubbles get TOO wide */
        }
        /* --- End Custom Alignment --- */
    """
) as iface:
    # 1. Improved, centered HTML heading
    gr.Markdown(value="<h1 style='text-align: center; margin-bottom: 20px;'>LetSetGo Chatbot</h1>")

    # 2. The main chatbot window
    # We give this row a specific ID ("chatbot-row") to apply the height CSS
    with gr.Row(elem_id="chatbot-row"):
        chatbot = gr.Chatbot(
            label="Chat",
            bubble_full_width=False,  # This is important
            render_markdown=True,  # Critical for tables & <details>
            elem_id="chatbot",  # Assign ID for potential future styling
            layout="panel"  # Adds a subtle border
        )

    # 3. The user input textbox
    with gr.Row():
        msg_box = gr.Textbox(
            show_label=False,
            placeholder="e.g., Show me the top 3 highest paid employees.",
            container=False,
            scale=8  # Give 8 parts of the space to the textbox
        )
        submit_btn = gr.Button(
            "Send",
            variant="primary",
            scale=1,  # Give 1 part of the space to the button
            min_width=70  # Give the button a minimum width so it doesn't look tiny
        )

    # 4. The examples
    gr.Examples(
        examples=[
            ["How many employees are there?"],
            ["Show me the top 3 highest paid employees."],
            ["What is the total salary of the 'Management' department?"],
            ["Give me all employees in 'Engineer'."],
            ["What is the salary of Deep?"]
        ],
        inputs=msg_box
    )

    # --- This is the logic that connects the UI ---
    # (This part is unchanged and correct)

    submit_btn.click(
        fn=chat_respond,
        inputs=[msg_box, chatbot],
        outputs=[chatbot, msg_box]
    )

    msg_box.submit(
        fn=chat_respond,
        inputs=[msg_box, chatbot],
        outputs=[chatbot, msg_box]
    )

if __name__ == "__main__":
    iface.launch()  # share=True