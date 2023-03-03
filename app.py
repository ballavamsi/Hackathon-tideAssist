from gpt_index import (
    SimpleDirectoryReader,
    GPTSimpleVectorIndex,
    LLMPredictor,
    PromptHelper,
)
from dotenv import load_dotenv
from pathlib import Path
from langchain import OpenAI
import os
import gradio as gr
import json
import hashlib
import logging
from langchain.cache import SQLiteCache
import langchain
from langchain.agents import Tool, initialize_agent

log = logging.getLogger("tideAssistLogger")
log.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

fh = logging.FileHandler("./conf/app.log", encoding="utf-8")
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
log.addHandler(fh)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
log.addHandler(ch)


AGENT = None
AI_INDEX = None
CACHE_FILE = "./conf/cache.json"
ENV_FILE = "./conf/.env"
INDEX_FILE = "./conf/index-{source}.json"
GRADIO_CSS = "./conf/gradio.custom.css"
CHECKSUMS_FILE = "./conf/checksums.json"
DATA_FOLDER = "./data"
DB_FILE = "./conf/.cache.db"

cache = {}
DOCUMENT_SOUCRE = "local"

AI_MODEL_NAME = "text-ada-001"
AI_TEMPERATURE = "0"


load_dotenv(dotenv_path=Path(ENV_FILE))
langchain.llm_cache = SQLiteCache(database_path=DB_FILE)


def construct_index():
    directory_path = DATA_FOLDER
    log.debug("Constructing index")
    max_input_size = 4096
    num_outputs = 1000
    max_chunk_overlap = 20
    chunk_size_limit = 2000

    log.debug("Defining LLM: " + AI_MODEL_NAME)
    llm_predictor = LLMPredictor(
        llm=OpenAI(
            temperature=AI_TEMPERATURE,
            model_name=AI_MODEL_NAME,
            max_tokens=num_outputs,
        )
    )
    prompt_helper = PromptHelper(
        max_input_size,
        num_outputs,
        max_chunk_overlap,
        chunk_size_limit=chunk_size_limit,
    )

    log.debug("Loading data")

    local_documents = SimpleDirectoryReader(directory_path).load_data()
    documents = local_documents

    log.debug("Indexing data")
    index = GPTSimpleVectorIndex(
        documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper
    )

    log.debug("Saving index")
    index.save_to_disk(INDEX_FILE.format(source=DOCUMENT_SOUCRE))
    return index


def load_index():
    log.debug("Loading index")
    global AI_INDEX
    if AI_INDEX is None and os.path.exists(INDEX_FILE.format(source=DOCUMENT_SOUCRE)):
        log.debug("Index not loaded, loading from disk")
        AI_INDEX = GPTSimpleVectorIndex.load_from_disk(
            INDEX_FILE.format(source=DOCUMENT_SOUCRE)
        )

    global AGENT
    if AGENT is None:
        tools = [
            Tool(name="QueryIndex", description="Query the index", func=query_index)
        ]

        llm = OpenAI(temperature=0)
        AGENT = initialize_agent(
            tools=tools, llm=llm, agent="zero-shot-react-description", verbose=True
        )


def save_cache(cache):
    log.debug("Saving cache")
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=4)


# Function to load the cache from a file
def load_cache():
    log.debug("Loading cache")
    with open(CACHE_FILE, "r") as f:
        return json.load(f)


if os.path.exists(CACHE_FILE):
    log.debug("Cache file exists using it")
    cache = load_cache()


def clean_text(text):
    return text.replace("Â", "").strip("\n").replace("\n", "<br/>")


def query_agent(text):
    log.debug("Querying agent")
    return AGENT.run(input=text)


def query_index(text):
    log.debug("Querying index")
    response = AI_INDEX.query(text, response_mode="tree_summarize")
    log.debug(response.source_nodes)
    return response.response


def ask_ai(state, text):
    log.debug("Asking AI")

    logging.info("User: " + text)
    answer = ""
    if text in cache:
        answer = cache[text]
        logging.info("AI: (cache)" + answer)
    else:
        answer = query_index(text)
        logging.info("AI: " + answer)

    answer = clean_text(answer)
    state = state + [(text, answer)]
    if text not in cache:
        cache[text] = answer
        save_cache(cache)
    return state, state


def construct_ui():
    log.debug("Starting gradio demo blocks")
    with gr.Blocks(css=GRADIO_CSS, title="tideAssist") as demo:

        gr.HTML(
            "<h1 class='title'>tide<span class='secondary-color'><b>Assist</b></span></h1>"
        )
        log.debug("Added logo")

        initial_value = [
            (
                None,
                "Hi, I am TideAssist. Ask me anything about Tide's plans and pricing.",
            )
        ]

        log.debug("Added initial value")
        chatbot = gr.Chatbot(
            show_label=False,
            elem_id="tideAssist",
            value=initial_value,
        )
        chatbot.style(color_map=(["#4852D8", "#242FAA"]))

        log.debug("Added chatbot")
        state = gr.State(initial_value)
        log.debug("Added state")

        with gr.Row(elem_id="chatbotRow"):
            with gr.Column(scale=1):
                txt = gr.Textbox(
                    elem_id="tideAssistInput",
                    show_label=False,
                    placeholder="Hey I'm tideAssist, ask me anything!",
                ).style(container=False)

        log.debug("Added textbox")
        txt.submit(ask_ai, [state, txt], [state, chatbot])
        txt.submit(lambda: "", None, txt)
    return demo


def content_changed():
    log.debug("Checking if content changed")
    checksums = {}
    for file in os.listdir(DATA_FOLDER):
        with open(os.path.join(DATA_FOLDER, file), "rb") as f:
            checksums[file] = hashlib.md5(f.read()).hexdigest()

    if not os.path.exists(INDEX_FILE.format(source=DOCUMENT_SOUCRE)):
        # if there is index file, we assume that the content has changed
        with open(CHECKSUMS_FILE, "w") as f:
            json.dump(checksums, f, indent=4)
        return True

    if os.path.exists(CHECKSUMS_FILE):
        with open(CHECKSUMS_FILE, "r") as f:
            old_checksums = json.load(f)
            if checksums == old_checksums:
                log.debug("Content unchanged")
                return False

    log.debug("Content changed")
    with open(CHECKSUMS_FILE, "w") as f:
        json.dump(checksums, f, indent=4)
    return True


if __name__ == "__main__":
    if content_changed():
        construct_index()
    load_index()
    construct_ui().launch(
        server_port=7173, share=True, favicon_path="./conf/favicon.png"
    )
