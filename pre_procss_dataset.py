import torch
from datasets import load_dataset
from transformers import AutoTokenizer

#eos and bos both map to the same id via AutoTokenizer.from_pretrained("gpt2") 
#So doesnt matter which one I use
def tokenize_function(examples):
    tokens = tokenizer(examples["text"], add_special_tokens=False)
    tokens["input_ids"] = [[eos_id] + ids + [eos_id] for ids in tokens["input_ids"]]
    return tokens

def group_texts(examples):
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated["input_ids"])
    total_length = (total_length // block_size) * block_size #This line exists to truncate the total length to a multiple of block_size

    result = {
        k: [t[i:i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

dataset = load_dataset("Salesforce/wikitext", "wikitext-103-v1")

train = dataset["train"]
test = dataset["test"]
valid = dataset["validation"]

##Clearing out the empty Strings 
train = train.filter(lambda x: x["text"].strip() != "")
test = test.filter(lambda x: x["text"].strip() != "")
valid = valid.filter(lambda x: x["text"].strip() != "")

tokenizer = AutoTokenizer.from_pretrained("gpt2")
eos_id = tokenizer.eos_token_id

tokenized_train = train.map(tokenize_function, batched=True, remove_columns=["text"]) #This line took me almost 5 minutes to run
tokenized_test = test.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_valid = valid.map(tokenize_function, batched=True, remove_columns=["text"])

block_size = 256 #This value need to match the one in the model declaration, and will by consequence be the context window size

##This is important to take the dataset from sentences of arbitrary token length to blocks of fixed size
lm_train_dataset = tokenized_train.map(group_texts, batched=True)
lm_test_dataset = tokenized_test.map(group_texts, batched=True)
lm_valid_dataset = tokenized_valid.map(group_texts, batched=True)


##This two lines also must match the model's specifications
vocab_size = tokenizer.vocab_size
embed_dim = 768

x_train_tensor = torch.tensor(lm_train_dataset["input_ids"])
y_train_tensor = torch.tensor(lm_train_dataset["labels"])

x_test_tensor = torch.tensor(lm_test_dataset["input_ids"])
y_test_tensor = torch.tensor(lm_test_dataset["labels"])

x_valid_tensor = torch.tensor(lm_valid_dataset["input_ids"])
y_valid_tensor = torch.tensor(lm_valid_dataset["labels"])

print(x_test_tensor[0:10])
print(y_train_tensor[0:10])

torch.save({
    'x_train': x_train_tensor,
    'y_train': y_train_tensor,
    'x_test': x_test_tensor,
    'y_test': y_test_tensor,
    'x_valid': x_valid_tensor,
    'y_valid': y_valid_tensor,
}, 'wikitext_tensors.pt')

print("Tensors saved to wikitext_tensors.pt")