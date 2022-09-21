from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
import argparse

tokenizer = GPT2Tokenizer.from_pretrained("models")
model = TFGPT2LMHeadModel.from_pretrained("models")

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", type=str, default="The quick brown fox")
parser.add_argument("--length", type=int, default=100)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--top_k", type=int, default=0)
parser.add_argument("--top_p", type=float, default=0.9)
args = parser.parse_args()

input_ids = tokenizer.encode(args.prompt, return_tensors='tf')
sample_outputs = model.generate(
    input_ids,
    do_sample=True,
    max_length=args.length,
    top_k=args.top_k,
    top_p=args.top_p,
    temperature=args.temperature,
    num_return_sequences=3
)

print("Output:", tokenizer.decode(sample_outputs[0], skip_special_tokens=True))
