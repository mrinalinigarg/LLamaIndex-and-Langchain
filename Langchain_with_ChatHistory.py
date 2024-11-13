from langchain_huggingface import HuggingFaceEndpoint

# Set your Hugging Face API token 
huggingfacehub_api_token = 'hf_token'

# Define the LLM
llm = HuggingFaceEndpoint(repo_id='tiiuae/falcon-7b-instruct', huggingfacehub_api_token=huggingfacehub_api_token)

# Predict the words following the text in question
question = 'Whatever you do, take care of your shoes'
output = llm.invoke(question)

print(output)

# Define the LLM
llm = OpenAI(model="gpt-3.5-turbo-instruct", api_key="<OPENAI_API_TOKEN>")

# Predict the words following the text in question
question = 'Whatever you do, take care of your shoes'
output = llm.invoke(question)

print(output)

# Create a prompt template from the template string
template = "You are an artificial intelligence assistant, answer the question. {question}"
prompt = PromptTemplate(template=template, input_variables=["question"])

# Create a chain to integrate the prompt template and LLM
llm = HuggingFaceEndpoint(repo_id='tiiuae/falcon-7b-instruct', huggingfacehub_api_token=huggingfacehub_api_token)
llm_chain = prompt | llm

question = "How does LangChain make LLM application development easier?"
print(llm_chain.invoke({"question": question}))

# Define an OpenAI chat model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key='<OPENAI_API_TOKEN>')		

# Create a chat prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "Respond to question: {question}")
    ]
)

# Chain the prompt template and model, and invoke the chain
llm_chain = prompt_template | llm
response = llm_chain.invoke({"question": "How can I retain learning?"})
print(response.content)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key='<OPENAI_API_TOKEN>')

# Create the conversation history and add the first AI message
history = ChatMessageHistory()
history.add_ai_message("Hello! Ask me anything about Python programming!")

# Add the user message to the history
history.add_user_message("What is a list comprehension?")

# Add another user message and call the model
history.add_user_message("Describe the same in fewer words")
response = llm.invoke(history.messages)
print(response.content)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key='<OPENAI_API_TOKEN>')

# Define a buffer memory
memory = ConversationBufferMemory(size=4)

# Define the chain for integrating the memory with the model
buffer_chain = ConversationChain(llm=llm, memory = memory)

# Invoke the chain with the inputs provided
buffer_chain.invoke("Write Python code to draw a scatter plot.")
buffer_chain.invoke("Use the Seaborn library.")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key='<YOUR_API_KEY_HERE>')

# Define a summary memory that uses an OpenAI chat model
memory = ConversationSummaryMemory(llm=llm)

# Define the chain for integrating the memory with the model
summary_chain = ConversationChain(llm=llm, memory = memory, verbose= True)
# Invoke the chain with the inputs provided
summary_chain.invoke("Describe the relationship of the human mind with the keyboard when taking a great online class.")
summary_chain.invoke("Use an analogy to describe it.")