import torch
import tiktoken
import gradio as gr
from torch.nn import functional as F
from transformer_model import GPTLanguageModel

enc = tiktoken.get_encoding("gpt2")
ckpt = torch.load("gpt_excuses.pt", map_location="cpu")
m = GPTLanguageModel(**ckpt["config"])
m.load_state_dict(ckpt["model_state"])
m.eval()
block_size = ckpt["config"]["block_size"]


@torch.no_grad()
def excuse(debut="", temperature=0.8, top_k=50, max_new_tokens=80):
    idx = torch.tensor([[198] + enc.encode(debut)])
    for _ in range(max_new_tokens):
        logits, _ = m(idx[:, -block_size:])
        logits = logits[:, -1, :] / temperature
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[:, [-1]]] = -float('inf')
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        if idx_next.item() == 198:
            break
        idx = torch.cat((idx, idx_next), dim=1)
    return enc.decode(idx[0, 1:].tolist())


def generer(debut):
    return excuse(debut, temperature=0.7)


demo = gr.Interface(
    fn=generer,
    inputs=gr.Textbox(label="Début de l'excuse (facultatif)", placeholder="Je suis en retard parce que"),
    outputs=gr.Textbox(label="Ton excuse"),
    title="Le Générateur d'Excuses Bidons",
    description="Un petit GPT entraîné de zéro sur 10 000 excuses absurdes. Aucune garantie que ton prof y croie.",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
