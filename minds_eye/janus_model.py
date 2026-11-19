"""Load Janus-Pro and run its three paths with an optional residual steer.

Janus-Pro is one autoregressive LLM (30 Llama-style blocks, d=4096) with two
heads: the ordinary text LM head, and ``gen_head``, which maps the same hidden
states to a 16384-entry VQ image codebook. Text and image generation therefore
share every transformer block, which is what makes the experiment possible: a
vector added to the residual stream in "language" is seen by the image head.

Three entry points:

- ``last_token_acts``: text-only forward, final-token residual at every layer.
  Used to build concept directions.
- ``draw``: text prompt -> 576 VQ tokens -> 384x384 images, with classifier-free
  guidance exactly as in the reference ``generation_inference.py``.
- ``write``: ordinary text generation, optionally returning the hidden states at
  every generated position (for Run 2).

Steering is a forward hook on one decoder block that adds ``scale * direction``
at every position of every forward pass while it is registered.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

_SHIMS = str(Path(__file__).resolve().parent.parent / "shims")
if _SHIMS not in sys.path:
    sys.path.insert(0, _SHIMS)

MODEL_ID = "deepseek-ai/Janus-Pro-7B"
N_IMG_TOKENS = 576
IMG_SIZE = 384
PATCH = 16


class Janus:
    def __init__(self, model_id: str = MODEL_ID, device: str = "cuda:0"):
        from transformers import AutoModelForCausalLM
        from janus.models import MultiModalityCausalLM, VLChatProcessor  # noqa: F401

        self.device = device
        self.proc = VLChatProcessor.from_pretrained(model_id)
        self.tok = self.proc.tokenizer
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
        self.model = model.to(torch.bfloat16).to(device).eval()
        self.lm = self.model.language_model
        self.layers = self.lm.model.layers
        self.n_layers = len(self.layers)
        self.d_model = self.lm.config.hidden_size

    # ---------------------------------------------------------------- prompts
    def chat_prompt(self, user: str, assistant: str = "") -> str:
        conv = [
            {"role": "<|User|>", "content": user},
            {"role": "<|Assistant|>", "content": assistant},
        ]
        return self.proc.apply_sft_template_for_multi_turn_prompts(
            conversations=conv, sft_format=self.proc.sft_format, system_prompt=""
        )

    # --------------------------------------------------------------- steering
    @contextmanager
    def steer(self, layer: int | None, direction, scale: float, rows=None, prompt_only: bool = False):
        """Add ``scale * direction`` to the output of block ``layer``.

        ``rows`` optionally restricts the add to a subset of batch rows (a bool
        tensor or index list); used to steer only the conditional half of a CFG
        batch. ``prompt_only`` adds the vector only on the prefill pass (the
        text prompt positions, seq len > 1) and leaves every cached decode step
        (each generated image token) unsteered, so the concept enters through
        attention to the prompt rather than as a per-position bias. ``layer=None`` or ``scale=0`` is a no-op, so call sites need no
        branching for the unsteered condition.
        """
        if layer is None or scale == 0.0 or direction is None:
            yield
            return
        vec = torch.as_tensor(np.asarray(direction), dtype=torch.float32)
        vec = vec.to(self.device, dtype=torch.bfloat16)

        def hook(_module, _inputs, output):
            hs = output[0] if isinstance(output, tuple) else output
            if prompt_only and hs.shape[1] == 1:
                return output  # cached decode step: an image/text token, leave it
            if rows is None:
                hs = hs + scale * vec
            else:
                hs = hs.clone()
                hs[rows] = hs[rows] + scale * vec
            if isinstance(output, tuple):
                return (hs,) + tuple(output[1:])
            return hs

        handle = self.layers[layer].register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    # ------------------------------------------------------------ activations
    @torch.inference_mode()
    def last_token_acts(self, texts: list[str], raw: bool = False) -> np.ndarray:
        """Final-token residual after every block: array (N, n_layers, d).

        ``raw=False`` wraps each text as a user turn and reads the state at the
        end of the assistant header, i.e. the state the model is in when it is
        about to respond to that text.
        """
        out = []
        for t in texts:
            prompt = t if raw else self.chat_prompt(t)
            ids = torch.tensor([self.tok.encode(prompt)], device=self.device)
            o = self.lm.model(input_ids=ids, output_hidden_states=True)
            hs = torch.stack(o.hidden_states[1:], dim=0)[:, 0, -1, :]  # (L, d)
            out.append(hs.float().cpu().numpy())
        return np.stack(out)

    # ----------------------------------------------------------------- images
    @torch.inference_mode()
    def draw(
        self,
        user_prompt: str,
        n: int = 8,
        seed: int = 0,
        cfg: float = 5.0,
        temperature: float = 1.0,
        layer: int | None = None,
        direction=None,
        scale: float = 0.0,
        steer_uncond: bool = True,
        return_tokens: bool = False,
        prompt_only: bool = False,
        return_hidden: bool = False,
    ):
        """Generate ``n`` images for one prompt. Returns uint8 (n, 384, 384, 3).

        ``return_hidden``: also return the final-layer hidden state of the
        first CONDITIONAL row at each of the 576 steps, (576, d). Run 2's
        positive control: these are in-regime states, so ``states_to_image``
        on them must give a picture.

        CFG batch layout follows the reference script: even rows conditional,
        odd rows unconditional (prompt body replaced by pad tokens). By default
        the steer is applied to every row: the steer is a change to the model's
        state, not to the prompt. Because the guided logit is
        ``u + cfg * (c - u)``, a shift common to both rows passes through once
        rather than being amplified by ``cfg``. ``steer_uncond=False`` steers
        only the conditional rows, which lets guidance amplify the steer; it is
        reported separately and is not the primary condition.
        """
        gen = torch.Generator(device=self.device).manual_seed(seed)
        prompt = self.chat_prompt(user_prompt) + self.proc.image_start_tag
        ids = torch.tensor(self.tok.encode(prompt), dtype=torch.long)
        tokens = torch.zeros((2 * n, len(ids)), dtype=torch.long, device=self.device)
        for i in range(2 * n):
            tokens[i, :] = ids
            if i % 2 != 0:
                tokens[i, 1:-1] = self.proc.pad_id
        embeds = self.lm.get_input_embeddings()(tokens)
        rows = None
        if not steer_uncond:
            rows = torch.arange(2 * n, device=self.device) % 2 == 0

        out_tokens = torch.zeros((n, N_IMG_TOKENS), dtype=torch.long, device=self.device)
        past = None
        hid_sink = []
        with self.steer(layer, direction, scale, rows=rows, prompt_only=prompt_only):
            for i in range(N_IMG_TOKENS):
                o = self.lm.model(inputs_embeds=embeds, use_cache=True, past_key_values=past)
                past = o.past_key_values
                if return_hidden:
                    hid_sink.append(o.last_hidden_state[0, -1, :].float().cpu().numpy())
                logits = self.model.gen_head(o.last_hidden_state[:, -1, :]).float()
                if cfg == 1.0:
                    guided = logits[0::2]
                else:
                    guided = logits[1::2] + cfg * (logits[0::2] - logits[1::2])
                probs = torch.softmax(guided / temperature, dim=-1)
                nxt = torch.multinomial(probs, 1, generator=gen)
                out_tokens[:, i] = nxt.squeeze(-1)
                both = torch.cat([nxt, nxt], dim=1).view(-1)
                embeds = self.model.prepare_gen_img_embeds(both).unsqueeze(1)

        imgs = self.decode_tokens(out_tokens)
        extras = []
        if return_tokens:
            extras.append(out_tokens.cpu().numpy())
        if return_hidden:
            extras.append(np.stack(hid_sink))
        return (imgs, *extras) if extras else imgs

    @torch.inference_mode()
    def decode_tokens(self, tokens) -> np.ndarray:
        """VQ token grid (n, 576) -> uint8 images (n, 384, 384, 3)."""
        tokens = torch.as_tensor(tokens, device=self.device).to(torch.int)
        n = tokens.shape[0]
        g = IMG_SIZE // PATCH
        dec = self.model.gen_vision_model.decode_code(tokens, shape=[n, 8, g, g])
        dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
        return np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------- text
    @torch.inference_mode()
    def write(
        self,
        user_prompt: str,
        max_new_tokens: int = 200,
        seed: int = 0,
        temperature: float = 0.7,
        layer: int | None = None,
        direction=None,
        scale: float = 0.0,
        return_hidden: bool = False,
        prompt_only: bool = False,
        stop_on_eos: bool = True,
        suppress_eos: bool = False,
    ):
        """Generate text. With ``return_hidden`` also return the final-layer
        (post-norm) hidden state at every generated position, shape (T, d):
        exactly the tensor ``gen_head`` would read if the model were drawing.

        ``stop_on_eos=False`` keeps generating through end-of-sequence so that
        exactly ``max_new_tokens`` hidden states come back (Run 2 needs a full
        576-position grid). The first EOS index is recorded in ``self.last_eos``
        (-1 if none) so a caller can report how much of the grid is post-EOS.
        ``suppress_eos=True`` instead masks the EOS logit so the model never
        ends and keeps writing the same essay (no restart junk in the grid)."""
        torch.manual_seed(seed)
        ids = torch.tensor([self.tok.encode(self.chat_prompt(user_prompt))], device=self.device)
        past = None
        cur = ids
        new_ids, hiddens = [], []
        self.last_eos = -1
        with self.steer(layer, direction, scale, prompt_only=prompt_only):
            for step in range(max_new_tokens):
                o = self.lm.model(input_ids=cur, use_cache=True, past_key_values=past)
                past = o.past_key_values
                h = o.last_hidden_state[:, -1, :]
                logits = self.lm.lm_head(h).float()
                if suppress_eos:
                    logits[:, self.tok.eos_token_id] = float("-inf")
                if temperature <= 0:
                    nxt = logits.argmax(-1, keepdim=True)
                else:
                    nxt = torch.multinomial(torch.softmax(logits / temperature, -1), 1)
                tid = int(nxt[0, 0])
                if tid == self.tok.eos_token_id:
                    if self.last_eos < 0:
                        self.last_eos = step
                    if stop_on_eos:
                        break
                new_ids.append(tid)
                if return_hidden:
                    hiddens.append(h[0].float().cpu().numpy())
                cur = nxt
        text = self.tok.decode(new_ids, skip_special_tokens=True)
        self.last_ids = list(new_ids)  # token ids aligned with the returned hiddens
        if return_hidden:
            return text, np.stack(hiddens) if hiddens else np.zeros((0, self.d_model))
        return text

    @torch.inference_mode()
    def read_states(self, ids, start: int, n: int = N_IMG_TOKENS) -> np.ndarray:
        """Reading regime (Run 5): ONE prefill forward over ``ids`` (a full token
        list incl. BOS and role headers), returning the final-layer (post-norm)
        states at positions ``start .. start+n-1``, shape (n, d). Teacher-forced:
        the state at position p is the one that predicts ``ids[p+1]``, so passing
        ``start = (index of the first essay token) - 1`` gives the same alignment
        as ``write(return_hidden=True)`` (whose k-th state predicted essay token k)."""
        t = torch.tensor([list(ids)], device=self.device)
        o = self.lm.model(input_ids=t)
        h = o.last_hidden_state[0, start:start + n, :]
        assert h.shape[0] == n, (h.shape, start, len(ids))
        return h.float().cpu().numpy()

    @torch.inference_mode()
    def states_to_image(self, hiddens, sample: bool = False, seed: int = 0,
                        temperature: float = 1.0):
        """Run 2 decode: final-layer hidden states (576, d) from ANY regime ->
        gen_head -> VQ tokens -> one 384x384 image. This is the model's own
        state->image map, unchanged from ``draw``; nothing is trained. Argmax by
        default; ``sample=True`` draws from the softmax as ``draw`` does (no CFG
        here: there is no unconditional row for a text pass)."""
        h = torch.as_tensor(np.asarray(hiddens), device=self.device, dtype=torch.bfloat16)
        assert h.shape[0] == N_IMG_TOKENS, h.shape
        logits = self.model.gen_head(h).float()
        if sample:
            gen = torch.Generator(device=self.device).manual_seed(seed)
            probs = torch.softmax(logits / temperature, dim=-1)
            toks = torch.multinomial(probs, 1, generator=gen).squeeze(-1)
        else:
            toks = logits.argmax(-1)
        return self.decode_tokens(toks.unsqueeze(0))[0], toks.cpu().numpy()


def save_grid(imgs: np.ndarray, path: str | os.PathLike, cols: int = 8, labels=None,
              tiles_dir: str | os.PathLike | None = None) -> None:
    """Write a contact sheet of ``imgs`` (n, H, W, 3).

    ``tiles_dir``: also write each image as a clean, unlabeled PNG
    (``00.png``, ``01.png``, ...) for figures. The sheet keeps the labels.
    """
    from PIL import Image, ImageDraw

    n, h, w, _ = imgs.shape
    rows = (n + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w, rows * h), "white")
    if tiles_dir is not None:
        Path(tiles_dir).mkdir(parents=True, exist_ok=True)
    for i in range(n):
        im = Image.fromarray(imgs[i])
        if tiles_dir is not None:
            im.save(Path(tiles_dir) / f"{i:02d}.png")
        if labels is not None:
            ImageDraw.Draw(im).text((4, 4), str(labels[i]), fill=(255, 255, 0))
        sheet.paste(im, ((i % cols) * w, (i // cols) * h))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
