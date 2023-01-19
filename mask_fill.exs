# Install all of our dependencies
Mix.install([
  {:bumblebee, "~> 0.1.2"},
  {:nx, "~> 0.4.1"},
  {:exla, "~> 0.4.1"},
  {:axon, "~> 0.3.1"},
])

# compile our numerical functions with EXLA
Nx.global_default_backend(EXLA.Backend)

# Load up a hugging face ML model. This one fills in missing words: https://huggingface.co/bert-base-uncased
{:ok, bert} = Bumblebee.load_model({:hf, "bert-base-uncased"})
{:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "bert-base-uncased"})

# Use Bumblebee to create a task definition for the model. in this case it is fill_mask
serving = Bumblebee.Text.fill_mask(bert, tokenizer)


# give a text input with a mask to fill
text = IO.gets("ENTER MASKED TEXT INPUT. ex. \"The Capital of [MASK] is Paris.\"\n")
# execute a one off run of the data and output predictions
output = Nx.Serving.run(serving, text)
IO.inspect(output)
