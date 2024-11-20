import gradio as gr
from sentence_transformers import SentenceTransformer, util

# Load the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

def calculate_similarity(source_sentence, sentences):
    if not source_sentence.strip():
        return "Please provide a source sentence."
    
    if not sentences.strip():
        return "Please provide a list of sentences."
    
    # Split the input sentences into a list
    sentence_list = [s.strip() for s in sentences.split("\n") if s.strip()]
    
    # Encode the source and the list of sentences
    source_embedding = model.encode(source_sentence, convert_to_tensor=True)
    sentence_embeddings = model.encode(sentence_list, convert_to_tensor=True)
    
    # Calculate similarities
    similarities = util.cos_sim(source_embedding, sentence_embeddings)
    results = [
        {"Sentence": sentence, "Similarity": round(sim.item(), 4)}
        for sentence, sim in zip(sentence_list, similarities[0])
    ]
    
    # Sort by similarity (highest first)
    results = sorted(results, key=lambda x: x["Similarity"], reverse=True)
    
    return results

# Define the Gradio interface
iface = gr.Interface(
    fn=calculate_similarity,
    inputs=[
        gr.Textbox(label="Source Sentence", placeholder="Enter the source sentence here"),
        gr.Textbox(label="List of Sentences", placeholder="Enter one sentence per line here", lines=5)
    ],
    outputs=gr.JSON(label="Similarities")
)

# Launch the app
iface.launch()
