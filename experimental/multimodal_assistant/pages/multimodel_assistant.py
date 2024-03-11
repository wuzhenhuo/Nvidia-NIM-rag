
import os
import base64
import pandas as pd
from PIL import Image
from io import BytesIO

import taipy.gui.builder as tgb
from taipy.gui import notify, State, invoke_long_callback, get_state_id

from bot_config.utils import get_config
from utils.memory import init_memory, get_summary, add_history_to_memory
from guardrails.fact_check import fact_check
from llm.llm_client import LLMClient
from retriever.embedder import NVIDIAEmbedders, HuggingFaceEmbeders
from retriever.vector import MilvusVectorClient, QdrantClient
from retriever.retriever import Retriever

from pages.knowledge_base import progress_for_logs, config, change_config, knowledge_base, mode, uploaded_file_paths

llm_client = LLMClient("mixtral_8x7b")

image_path = None
messages = [{"role": "assistant", "style": "assistant_message", "content": "Hi, what can I help you with?"}]
sources = {}
image_query = ""
current_user_message = ""
summary = "No summary yet."
messages_dict = {}
logs_for_messages = ""

vector_client = MilvusVectorClient(hostname="localhost", port="19530", collection_name=config["core_docs_directory_name"])
memory = init_memory(llm_client.llm, config['summary_prompt'])
query_embedder = NVIDIAEmbedders(name="nvolveqa_40k", type="query")
retriever = Retriever(embedder=query_embedder , vector_client=vector_client)


def on_image_upload(state):
    notify(state, "s", "Image loaded for multimodal RAG Q&A.") 
    neva = LLMClient("neva_22b")
    image = Image.open(state.image_path).convert("RGB")
    buffered = BytesIO()
    image.save(buffered, format="JPEG", quality=20) # Quality = 20 is a workaround (WAR)
    b64_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
    notify(state, "i", "Getting image description using NeVA...")
    res = neva.multimodal_invoke(b64_string, creativity = 0, quality = 9, complexity = 0, verbosity = 9)
    notify(state, "s", "Image description received!")
    state.image_query = res.content


def send_message_asynchronous(retriever, current_user_message, messages, state_id):   
    global progress_for_logs
    progress_for_logs[state_id] = {'message_logs': []}

    progress_for_logs[state_id]['message_logs'].append("Step 1/5: Sending message and getting relevant documents...")
    context, sources = retriever.get_relevant_docs(current_user_message)
    augmented_prompt = "Relevant documents:" + context + "\n\n[[QUESTION]]\n\n" + current_user_message #+ "\n" + config["footer"]
    system_prompt = config["header"]

    progress_for_logs[state_id]['message_logs'].append("Step 2/5: Responding based on documents...")
    response = llm_client.chat_with_prompt(system_prompt, augmented_prompt)
    full_response = "".join(response)

    progress_for_logs[state_id]['message_logs'].append("Step 3/5: Adding message to memory...")
    add_history_to_memory(memory, current_user_message, full_response)
    messages.append({"role": "assistant", "style": "assistant_message", "content": full_response})
    messages.append({"role": "assistant", "style": "assistant_info_message", "content": "Fact Check result:" })
    full_response += "\n\nFact Check result: "

    progress_for_logs[state_id]['message_logs'].append("Step 4/5: Running fact checking/guardrails...")
    res = fact_check(context, current_user_message, full_response)
    fact_check_response = "".join(res)
    right_or_wrong = fact_check_response.split(' ')[0]
    messages.append({"role": "assistant", 
                     "style": right_or_wrong.replace('\\', ''), 
                     "content": fact_check_response.replace(right_or_wrong, '')})

    progress_for_logs[state_id]['message_logs'].append("Step 5/5: Getting summary...")
    summary = get_summary(memory)
    return messages, summary, sources

def when_chat_answers(state, status, res):
    global progress_for_logs
    state.logs_for_messages = "\n".join(progress_for_logs[get_state_id(state)]['message_logs'])
    
    if isinstance(status, bool):
        state.messages, state.summary, state.sources = res
        state.current_user_message = ""
        state.image_query = ""
        state.image_path = None
        state.logs_for_messages = None
        state.conv.update_content(state, create_conv(state))
        notify(state, "success", "Response received!")

def send_message(state: State) -> None:
    notify(state, "info", "Sending message...")
    current_user_message = state.current_user_message
    state.messages.append({"role": "user", "style":"user_message", "content": current_user_message})
    state.current_user_message = ""

    if state.image_query:
        current_user_message = f"\nI have uploaded an image with the following description: {state.image_query}" + "Here is the question: " + current_user_message

    state.conv.update_content(state, create_conv(state))

    invoke_long_callback(state, 
                         send_message_asynchronous, [state.retriever, current_user_message, state.messages, get_state_id(state)],
                         when_chat_answers, [],
                         period=5000)



with tgb.Page() as multimodal_assistant:
    with tgb.layout("2 8", gap="50px"):
        with tgb.part("sidebar"):
            tgb.text("Assistant mode", class_name="h4")
            tgb.text("Select a configuration/type of bot")
            tgb.selector(value="{mode}", lov=["multimodal"], dropdown=True, class_name="fullwidth", on_change=change_config, label="Mode")
            
            tgb.html("hr")
            
            tgb.text("Image Input Query", class_name="h4")
            tgb.text("Upload an image (JPG/JPEG/PNG):")
            # TODO: Allow multiple images
            tgb.file_selector(content="{image_path}", on_action=on_image_upload,
                              extensions='.jpg,.jpeg,.png', 
                              label="Upload an image")

            tgb.html("hr")

            tgb.image('{image_path}', width="100%")

            with tgb.part(render='{image_query}'):
                tgb.text('Image Description', class_name="h5")
                tgb.text('{image_query}')

        with tgb.part():
            tgb.navbar()
            tgb.text("{config['page_title']}", class_name="h1")
            tgb.text("{config.instructions}", mode="md")

            with tgb.part(class_name="p1"): # UX concerns
                tgb.part(partial="{conv}", height="600px", class_name="card card_chat")

                tgb.text("{logs_for_messages}", mode="pre")

                with tgb.part("card mt1"): # UX concerns
                    tgb.input("{current_user_message}", on_action=send_message, change_delay=-1, label="Write your message:",  class_name="fullwidth")

            tgb.html("hr")

            tgb.text("Summary: {summary}", mode="md")



def create_conv(state):
    messages_dict = {}

    document_paths = list(state.sources.keys())
    # Get all the names of the files present in these paths
    document_names_to_download = [os.path.basename(document_path) for document_path in document_paths]
    with tgb.Page() as new_partial:
        for i, message in enumerate(state.messages):
            text = message["content"].replace("<br>", "\n").replace('"', "'")
            messages_dict[f"message_{i}"] = text
            tgb.text("{messages_dict['"+f"message_{i}"+"']}", class_name=message["style"], mode="md")
        with tgb.layout("1 1 1 1 1"):
            for i, document_path in enumerate(document_paths):
                tgb.file_download(content=document_path, name=document_names_to_download[i], label=document_names_to_download[i])
        
    state.messages_dict = messages_dict
    return new_partial
