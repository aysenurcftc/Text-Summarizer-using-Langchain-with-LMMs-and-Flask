from flask import Flask, request, jsonify, render_template
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain_community.llms import CTransformers
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')



def chunks_and_document(txt):
    text_splitter = CharacterTextSplitter()  
    texts = text_splitter.split_text(txt)  
    docs = [Document(page_content=t) for t in texts] 
    return docs


# Loading the Llama 2's LLM
def load_llm():
    # We instantiate the callback with a streaming stdout handler
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])   

    # Loading the LLM model
    llm = CTransformers(
        model="models/llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )
    return llm


# This function is used for applying the llm model to our document 
def chains_and_response(docs):
    llm = load_llm()
    chain = load_summarize_chain(llm, chain_type='map_reduce')
    return chain.invoke(docs)


@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.json
    #print("Received data:", data)  # Debugging line
    txt_input = data.get('text', '')
    #print("Input text:", txt_input)  # Debugging line
    
    if not txt_input:
        return jsonify({'error': 'No text provided'}), 400
    
    docs = chunks_and_document(txt_input)
    response = chains_and_response(docs)
    #print("Summary response:", response)  # Debugging line
    
    return jsonify({'summary': response['output_text']})




if __name__ == '__main__':
    app.run(debug=True)
