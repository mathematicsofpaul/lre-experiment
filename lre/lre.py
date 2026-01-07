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
        
    def get_hidden_state(self, text, layer_name, token_position="last"):
        """
        Runs the model and extracts the hidden state at a specific token position.
        
        Args:
            text: The input text
            layer_name: Which layer to extract from
            token_position: Either "last" (last token of entire prompt) or an integer index
        """
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        # Determine which token to extract from
        if token_position == "last":
            token_idx = inputs["input_ids"].shape[1] - 1
        else:
            token_idx = token_position

        with TraceDict(self.model, [layer_name]) as ret:
            self.model(**inputs)
            
        # Extract vector: [Batch, Seq, Hidden] -> [Hidden]
        # Handle both tuple output (e.g., GPT-2) and tensor output (e.g., Qwen)
        output = ret[layer_name].output
        if isinstance(output, tuple):
            # Output is a tuple, extract the first element
            h = output[0][0, token_idx, :].detach().cpu().numpy()
        else:
            # Output is already a tensor
            h = output[0, token_idx, :].detach().cpu().numpy()
        return h
    
    def find_subject_last_token(self, prompt, subject):
        """
        Find the last token position of the subject in the prompt using offset mapping.
        This avoids tokenization mismatch issues.
        
        Args:
            prompt: The full formatted prompt containing the subject
            subject: The subject string to find
            
        Returns:
            The token index of the last token of the subject's last occurrence
        """
        # Tokenize with offset mapping to get character-level positions
        inputs = self.tokenizer(prompt, return_tensors="pt", return_offsets_mapping=True)
        offset_mapping = inputs["offset_mapping"][0]  # Shape: (seq_len, 2)
        
        # Find the last occurrence of subject in the prompt
        subject_start_char = prompt.rfind(subject)
        if subject_start_char == -1:
            raise ValueError(f"Subject '{subject}' not found in prompt: '{prompt}'")
        
        subject_end_char = subject_start_char + len(subject)
        
        # Find which token contains the end of the subject
        # We want the last token that overlaps with the subject
        subject_last_token = None
        for token_idx, (start, end) in enumerate(offset_mapping):
            # Check if this token overlaps with the subject
            if start < subject_end_char and end > subject_start_char:
                subject_last_token = token_idx
        
        if subject_last_token is None:
            raise ValueError(f"Could not find subject '{subject}' in tokenized prompt")
        
        return subject_last_token

    def train_lre(self, training_data, layer_name, template, extract_from="subject_end"):
        """
        Learns the matrix W and bias b such that: W * h_subject + b ≈ h_object_output
        
        Args:
            training_data: List of samples with 'subject' and 'object'
            layer_name: Layer to extract hidden states from
            template: Prompt template with {} placeholder
            extract_from: "subject_end" (default) or "template_end"
                - "subject_end": Extract from last token of subject (e.g., "water")
                - "template_end": Extract from last token of full prompt (e.g., "with")
        """
        X = [] # Subject hidden states (inputs)
        Y = [] # Object target outputs (we will approximate the next token logits)

        for sample in training_data:
            subj = sample['subject']
            prompt = template.format(subj)
            
            # 1. Get Input State (s) at the specific layer
            if extract_from == "subject_end":
                # Use offset mapping to find subject's last token position
                subject_last_token_pos = self.find_subject_last_token(prompt, subj)
                h_s = self.get_hidden_state(prompt, layer_name, token_position=subject_last_token_pos)
            else:  # "template_end"
                # Extract from last token of entire prompt
                h_s = self.get_hidden_state(prompt, layer_name, token_position="last")
            
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

    def evaluate(self, lre_operator, test_data, layer_name, template, verbose=True, extract_from="subject_end"):
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

            # 1. Get current hidden state (matching training extraction point)
            if extract_from == "subject_end":
                # Use offset mapping to find subject's last token position
                subject_last_token_pos = self.find_subject_last_token(prompt, subj)
                h = self.get_hidden_state(prompt, layer_name, token_position=subject_last_token_pos)
            else:  # "template_end"
                h = self.get_hidden_state(prompt, layer_name, token_position="last")
            
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