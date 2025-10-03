import clip
import torch.nn as nn
import torch


class CoOp(nn.Module):
    def __init__(self, classnames, model_name="RN50", n_ctx=16, device="cuda"):
        super().__init__()
        model, preprocess = clip.load(model_name, device=device)
        self.base_model = model
        self.tokenizer = clip.tokenize
        self.classnames = classnames

        self.n_ctx = n_ctx
        self.ctx_dim = model.ln_final.weight.shape[0]
        self.dtype = model.dtype
        self.device = device

        self.base_model.eval()
        for p in self.base_model.parameters():
            p.requires_grad_(False)

        self.text_tokenize(classnames)

    def text_tokenize(self, classnames):
        ctx_init = "a photo of a"
        prompt_n_ctx = len(ctx_init.split(" "))
        prompt = self.tokenizer(ctx_init).to(self.device)
        with torch.no_grad():
            embedding = self.base_model.token_embedding(prompt)
        ctx_vectors = torch.zeros(self.n_ctx, self.ctx_dim, dtype=self.dtype)
        ctx_vectors[self.n_ctx - prompt_n_ctx :, :] = embedding[0, 1 : 1 + prompt_n_ctx, :]
        ctx_vectors = ctx_vectors.to(self.device)
        self.ctx = nn.Parameter(ctx_vectors)

        prompt_prefix = " ".join(["X"] * (self.n_ctx - prompt_n_ctx))
        prompt_prefix = f"{prompt_prefix} {ctx_init}"
        print(prompt_prefix)

        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([self.tokenizer(p) for p in prompts]).to(self.device)
        with torch.no_grad():
            embedding = self.base_model.token_embedding(tokenized_prompts)

        self.register_buffer("prefix", embedding[:, :1, :])
        self.register_buffer("suffix", embedding[:, 1 + self.n_ctx :, :])
        self.tokenized = tokenized_prompts

    def forward(self, images):
        image_features = self.base_model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        ctx = self.ctx.unsqueeze(0).expand(len(self.classnames), -1, -1)
        prompts = torch.cat([self.prefix, ctx, self.suffix], dim=1)

        x = prompts + self.base_model.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2).type(self.dtype)  # NLD -> LND
        x = self.base_model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.base_model.ln_final(x).type(self.dtype)

        text_features = x[torch.arange(x.shape[0]), self.tokenized.argmax(dim=-1)] @ self.base_model.text_projection
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.base_model.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        return logits


if __name__ == "__main__":
    model = CoOp(["car", "truck"], n_ctx=16, device="cuda:0")
    for p in model.parameters():
        if p.requires_grad:
            print(p.shape)
    imgs = torch.randn(10, 3, 224, 224).to("cuda:0")
    model = model.to("cuda:0")
    out = model(imgs)

    print(model)
