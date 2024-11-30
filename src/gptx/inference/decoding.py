import torch


class GenerationMixin(torch.nn.Module):
    # TODO: @liutong
    @torch.inference_mode()
    def generate(self, input_ids: torch.Tensor, stop_words, do_sample=False, top_k=0, top_p=1.0, max_new_tokens=16):
        self.eval()
        generated_token_ids = []
        while len(generated_token_ids) < max_new_tokens:
            logits = self(input_ids)[0]
            next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            if next_token in stop_words:
                break
        return input_ids