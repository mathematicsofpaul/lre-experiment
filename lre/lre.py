import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from baukit import TraceDict
from sklearn.linear_model import LinearRegression
import os

# Ensure progress bars are shown during model downloads
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '0'

class LREModel:
    def __init__(self, model_name="gpt2-xl", device="cpu", token=None):
        print(f"Loading {model_name} on {device}...")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            token=token,
            local_files_only=False  # Allow downloads and show progress
        ).to(device)
        self.model.eval()
        
        # GPT-2/J specific tokenization configs
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def get_hidden_state(self, text, layer_name):
        """
        Runs the model and extracts the hidden state at the last token of the prompt.
        This is where the model is about to predict the answer.
        """
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        # Extract from the last token of the entire prompt
        # This is where the model computes the next token prediction
        last_token_idx = inputs["input_ids"].shape[1] - 1

        with TraceDict(self.model, [layer_name]) as ret:
            self.model(**inputs)
            
        # Extract vector: [Batch, Seq, Hidden] -> [Hidden]
        # Handle both tuple output (e.g., GPT-2) and tensor output (e.g., Qwen)
        output = ret[layer_name].output
        if isinstance(output, tuple):
            # Output is a tuple, extract the first element
            h = output[0][0, last_token_idx, :].detach().cpu().numpy()
        else:
            # Output is already a tensor
            h = output[0, last_token_idx, :].detach().cpu().numpy()
        return h

    def train_lre(self, training_data, layer_name, template):
        """
        Learns the matrix W and bias b such that: W * h_subject + b ≈ h_object_output
        """
        X = [] # Subject hidden states (inputs)
        Y = [] # Object target outputs (we will approximate the next token logits)

        for sample in training_data:
            subj = sample['subject']
            prompt = template.format(subj)
            
            # 1. Get Input State (s) at the specific layer
            h_s = self.get_hidden_state(prompt, layer_name)
            X.append(h_s)

            # 2. Get the "Ideal" direction. 
            # In the original paper, they map s -> z (final layer output).
            # We approximate this by looking at the embedding of the target object.
            # Handle models that add special tokens (like Gemma's <bos>)
            encoded = self.tokenizer.encode(" " + sample['object'], add_special_tokens=False)
            if len(encoded) == 0:
                # Fallback: try without leading space
                encoded = self.tokenizer.encode(sample['object'], add_special_tokens=False)
            target_id = encoded[0]
            
            # Get the weight of the output embedding for the target token
            # This represents the direction in the final space corresponding to the answer
            target_vec = self.model.lm_head.weight[target_id].detach().cpu().numpy()
            Y.append(target_vec)

        # Solve for Linear mapping: Y = XW + b
        reg = LinearRegression().fit(X, Y)
        return reg

    def evaluate(self, lre_operator, test_data, layer_name, template, verbose=True):
        correct = 0
        total = 0
        
        # Reset eval_results for this evaluation
        self._eval_results = []
        
        print("\n" + "="*80)
        print(f"{'EVALUATION RESULTS':^80}")
        print("="*80)
        
        # Show sample prompt if verbose
        if verbose and len(test_data) > 0:
            sample_prompt = template.format(test_data[0]['subject'])
            print(f"\nSample prompt structure (first test item):")
            print(f"{'─'*80}")
            print(f"{sample_prompt}")
            print(f"{'─'*80}\n")
        
        print(f"{'Subject':<25} {'Expected':<15} {'LRE Prediction':<15} {'Status':>10}")
        print("-"*80)
        
        for sample in test_data:
            subj = sample['subject']
            expected = sample['object']
            prompt = template.format(subj)

            # 1. Get current hidden state
            h = self.get_hidden_state(prompt, layer_name)
            
            # 2. Apply Linear Transformation (LRE)
            # z_pred = W * h + b
            z_pred = lre_operator.predict([h])[0]  # Returns numpy array
            z_pred = torch.tensor(z_pred, dtype=torch.float32).to(self.device)

            # 3. Decode directly from this predicted vector (Rank 1 approximation)
            # We treat z_pred as if it were the final layer state entering the unembedding
            logits = torch.matmul(z_pred, self.model.lm_head.weight.T)
            probs = torch.softmax(logits, dim=0)
            
            top_token_id = torch.argmax(probs).item()
            predicted_word = self.tokenizer.decode(top_token_id).strip()

            is_match = predicted_word.lower().startswith(expected.lower())
            status = '✓ Correct' if is_match else '✗ Wrong'
            print(f"{subj:<25} {expected:<15} {predicted_word:<15} {status:>10}")
            
            if is_match: correct += 1
            total += 1 

            # Store prediction details for analysis
            self._eval_results.append({
                'subject': subj,
                'expected': expected,
                'predicted': predicted_word,
                'z_pred': z_pred.detach().cpu().numpy(),
                'top_token_id': top_token_id,
                'status': status
            })
        
        print("="*80)
        print(f"{'Faithfulness Score:':<40} {correct}/{total} ({correct/total:.2%})")
        print("="*80)
        
        return {
            'faithfulness': correct / total if total > 0 else 0,
            'correct': correct,
            'total': total, 
            'eval_results': getattr(self, '_eval_results', [])
        }