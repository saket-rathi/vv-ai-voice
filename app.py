import os
import gradio as gr
from gradio_client import Client, handle_file

# Global variable to store the client connection
client = None

def get_client():
    """
    Connects to the Hugging Face Space only when needed.
    This prevents the app from crashing during startup if the Space is sleeping.
    """
    global client
    if client is None:
        print("⏳ Connecting to Hugging Face Space...")
        HF_TOKEN = os.environ.get("HF_TOKEN")
        # Connect to your specific private/public space
        client = Client("saketrathi111/vibe-voice-custom-voices-api", token=HF_TOKEN)
        print("✅ Connected successfully!")
    return client

def generate_voice(text, audio_path):
    try:
        # 1. Initialize the client (wakes up the Space if needed)
        my_client = get_client()
        
        # 2. Prepare the audio file for upload
        audio_file = handle_file(audio_path)
        
        # 3. Send the request
        # Note: If your custom space uses different input names, update them here.
        # These are the standard ones from the original Vibe Voice.
        print(f"🚀 Sending text: '{text[:20]}...'")
        result = my_client.predict(
            text=text,
            speaker1_audio_path=audio_file,
            speaker2_audio_path=audio_file,
            speaker3_audio_path=audio_file,
            speaker4_audio_path=audio_file,
            seed=42,
            diffusion_steps=20,
            cfg_scale=1.3,
            use_sampling=False,
            temperature=0.95,
            top_p=0.95,
            max_words_per_chunk=250,
            api_name="/generate_speech_gradio"
        )
        return result
        
    except Exception as e:
        # This error message will show up in the output box if something breaks
        return f"Error: {str(e)}. (If this is a timeout, try clicking Generate again in 1 minute)"

# --- Frontend Layout ---
with gr.Blocks() as demo:
    gr.Markdown("# 🎙️ Custom Vibe Voice AI")
    gr.Markdown(
        "**Status:** The backend AI will wake up automatically when you use it. "
        "The first generation might take 2-3 minutes."
    )
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(label="Text to Speak", placeholder="Type something here...")
            input_audio = gr.Audio(label="Upload Reference Voice", type="filepath")
            btn = gr.Button("Generate Speech", variant="primary")
        
        with gr.Column():
            output_audio = gr.Audio(label="Generated Result")

    # Connect the button to the function
    btn.click(
        fn=generate_voice, 
        inputs=[input_text, input_audio], 
        outputs=output_audio
    )

# --- Render Startup Configuration ---
if __name__ == "__main__":
    # Render assigns a random port in the PORT env var. Default to 10000 if missing.
    port = int(os.environ.get("PORT", 10000))
    print(f"🚀 Starting app on port {port}...")
    
    # server_name="0.0.0.0" is CRITICAL for Render to see the app
    demo.launch(server_name="0.0.0.0", server_port=port, show_error=True)
