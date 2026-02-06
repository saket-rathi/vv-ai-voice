import os
import gradio as gr
from gradio_client import Client, handle_file

# This tells Python: "Look for a variable named HF_TOKEN. If you can't find it, use None."
HF_TOKEN = os.environ.get("HF_TOKEN")

# Initialize the client using that variable
client = Client("saketrathi111/vibe-voice-custom-voices-api", token=HF_TOKEN)

def generate_voice(text, audio_path):
    # The API requires 4 speaker paths. 
    # For a barebone app, we use your one upload for all 4 slots.
    audio_file = handle_file(audio_path)
    
    result = client.predict(
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

# Building the UI
with gr.Blocks() as demo:
    gr.Markdown("# 🎙️ Vibe Voice API Wrapper")
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(label="Text to Speak", placeholder="Type something here...")
            input_audio = gr.Audio(label="Upload Reference Voice", type="filepath")
            btn = gr.Button("Generate Speech", variant="primary")
        
        with gr.Column():
            output_audio = gr.Audio(label="Generated Result")

    # Link the button to the function
    btn.click(
        fn=generate_voice, 
        inputs=[input_text, input_audio], 
        outputs=output_audio
    )

# Keep all your other code the same...

# Change this part
app = demo

if __name__ == "__main__":
    # Render's default is 10000, but we use their PORT variable to be safe
    # We MUST bind to 0.0.0.0
    port = int(os.environ.get("PORT", 10000))
    
    print(f"Starting app on port {port}...")
    demo.launch(
        server_name="0.0.0.0", 
        server_port=port,
        show_error=True
    )
