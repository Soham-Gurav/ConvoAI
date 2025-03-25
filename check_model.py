import google.generativeai as genai

# Configure with your API key
genai.configure(api_key="")

# List all available models
models = genai.list_models()

# Print model names
for model in models:
    print(model.name)




