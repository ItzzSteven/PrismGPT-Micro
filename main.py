
import math
import random
import json

random.seed(42)

try:
    with open("input.txt", "r", encoding="utf-8") as f:
        docs = [
            d.strip()
            for d in f.read().split("\n\n")
            if d.strip()
        ]
except FileNotFoundError:
    print("ERROR: input.txt not found!")
    print("Create input.txt with training conversations.")
    raise SystemExit

if not docs:
    raise ValueError("input.txt contains no conversations.")

random.shuffle(docs)

print("Conversations:", len(docs))

uchars = sorted(set("".join(docs)))

BOS = len(uchars)
vocab_size = len(uchars) + 1

stoi = {ch: i for i, ch in enumerate(uchars)}

print("Vocabulary:", vocab_size)

class Value:

    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(
            self.data + other.data,
            (self, other),
            (1, 1)
        )

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(
            self.data * other.data,
            (self, other),
            (other.data, self.data)
        )

    def __pow__(self, other):
        return Value(
            self.data ** other,
            (self,),
            (other * self.data ** (other - 1),)
        )

    def log(self):
        return Value(
            math.log(self.data),
            (self,),
            (1 / self.data,)
        )

    def exp(self):
        return Value(
            math.exp(self.data),
            (self,),
            (math.exp(self.data),)
        )

    def relu(self):
        return Value(
            max(0, self.data),
            (self,),
            (float(self.data > 0),)
        )

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other ** -1

    def __rtruediv__(self, other):
        return other * self ** -1

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)

                for child in v._children:
                    build_topo(child)

                topo.append(v)

        build_topo(self)

        self.grad = 1

        for v in reversed(topo):
            for child, local_grad in zip(
                v._children,
                v._local_grads
            ):
                child.grad += local_grad * v.grad

n_embd = 64
n_head = 4
n_layer = 2
block_size = 128

head_dim = n_embd // n_head

def matrix(nout, nin, std=0.02):
    return [
        [Value(random.gauss(0, std)) for _ in range(nin)]
        for _ in range(nout)
    ]


state_dict = {
    "wte": matrix(vocab_size, n_embd),
    "wpe": matrix(block_size, n_embd),
    "lm_head": matrix(vocab_size, n_embd)
}

for i in range(n_layer):

    state_dict[f"layer{i}.attn_wq"] = matrix(n_embd, n_embd)
    state_dict[f"layer{i}.attn_wk"] = matrix(n_embd, n_embd)
    state_dict[f"layer{i}.attn_wv"] = matrix(n_embd, n_embd)
    state_dict[f"layer{i}.attn_wo"] = matrix(n_embd, n_embd, std=0)

    state_dict[f"layer{i}.mlp_fc1"] = matrix(4 * n_embd, n_embd)
    state_dict[f"layer{i}.mlp_fc2"] = matrix(n_embd, 4 * n_embd, std=0)


params = [
    p
    for mat in state_dict.values()
    for row in mat
    for p in row
]

print("Parameters:", len(params))

def linear(x, w):
    return [
        sum(wi * xi for wi, xi in zip(row, x))
        for row in w
    ]


def softmax(logits):
    max_val = max(v.data for v in logits)

    exps = [
        (v - max_val).exp()
        for v in logits
    ]

    total = sum(exps)

    return [e / total for e in exps]


def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5

    return [xi * scale for xi in x]

def gpt(token_id, pos_id, keys, values):

    tok_emb = state_dict["wte"][token_id]
    pos_emb = state_dict["wpe"][pos_id]

    x = [
        t + p
        for t, p in zip(tok_emb, pos_emb)
    ]

    x = rmsnorm(x)

    for li in range(n_layer):

        # Multi-head attention

        x_residual = x
        x = rmsnorm(x)

        q = linear(x, state_dict[f"layer{li}.attn_wq"])
        k = linear(x, state_dict[f"layer{li}.attn_wk"])
        v = linear(x, state_dict[f"layer{li}.attn_wv"])

        keys[li].append(k)
        values[li].append(v)

        x_attn = []

        for h in range(n_head):

            hs = h * head_dim

            q_h = q[hs:hs + head_dim]

            k_h = [
                ki[hs:hs + head_dim]
                for ki in keys[li]
            ]

            v_h = [
                vi[hs:hs + head_dim]
                for vi in values[li]
            ]

            attn_logits = [
                sum(
                    q_h[j] * k_h[t][j]
                    for j in range(head_dim)
                ) / head_dim ** 0.5
                for t in range(len(k_h))
            ]

            attn_weights = softmax(attn_logits)

            head_out = [
                sum(
                    attn_weights[t] * v_h[t][j]
                    for t in range(len(v_h))
                )
                for j in range(head_dim)
            ]

            x_attn.extend(head_out)

        x = linear(
            x_attn,
            state_dict[f"layer{li}.attn_wo"]
        )

        x = [
            a + b
            for a, b in zip(x, x_residual)
        ]

        # Feedforward network

        x_residual = x
        x = rmsnorm(x)

        x = linear(
            x,
            state_dict[f"layer{li}.mlp_fc1"]
        )

        x = [
            xi.relu() ** 2
            for xi in x
        ]

        x = linear(
            x,
            state_dict[f"layer{li}.mlp_fc2"]
        )

        x = [
            a + b
            for a, b in zip(x, x_residual)
        ]

    return linear(x, state_dict["lm_head"])

learning_rate = 1e-2
beta1 = 0.9
beta2 = 0.95
eps_adam = 1e-8

m = [0.0] * len(params)
v = [0.0] * len(params)

num_steps = 10000

print("\nStarting GPT training...\n")

for step in range(num_steps):

    doc = random.choice(docs)

    # Random training window for long conversations
    if len(doc) > block_size:
        start = random.randint(
            0,
            len(doc) - block_size
        )
        doc = doc[start:start + block_size]

    tokens = [
        BOS
    ] + [
        stoi[ch]
        for ch in doc
    ]

    n = min(block_size, len(tokens) - 1)

    keys = [[] for _ in range(n_layer)]
    values = [[] for _ in range(n_layer)]

    losses = []

    # Forward pass

    for pos_id in range(n):

        token_id = tokens[pos_id]
        target_id = tokens[pos_id + 1]

        logits = gpt(
            token_id,
            pos_id,
            keys,
            values
        )

        probs = softmax(logits)

        loss_t = -probs[target_id].log()

        losses.append(loss_t)

    loss = (1 / n) * sum(losses)

    # Backpropagation

    loss.backward()

    # Adam update

    lr_t = learning_rate * 0.5 * (
        1 + math.cos(math.pi * step / num_steps)
    )

    for i, p in enumerate(params):

        m[i] = beta1 * m[i] + (1 - beta1) * p.grad

        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2

        m_hat = m[i] / (1 - beta1 ** (step + 1))

        v_hat = v[i] / (1 - beta2 ** (step + 1))

        p.data -= (
            lr_t * m_hat /
            (v_hat ** 0.5 + eps_adam)
        )

        p.grad = 0

    if (step + 1) % 100 == 0:
        print(
            f"Step {step + 1}/{num_steps} "
            f"| Loss: {loss.data:.4f}"
        )

print("\nSaving model...")

model_weights = {
    k: [
        [vi.data for vi in row]
        for row in mat
    ]
    for k, mat in state_dict.items()
}

with open("model.json", "w", encoding="utf-8") as f:
    json.dump(model_weights, f)

print("Model saved!")

temperature = 0.7


def generate(prompt, max_tokens=300):

    # Use the latest context window
    prompt = prompt[-(block_size - 1):]

    keys = [[] for _ in range(n_layer)]
    values = [[] for _ in range(n_layer)]

    # Begin sequence
    logits = gpt(BOS, 0, keys, values)

    # Feed prompt characters
    for pos_id, ch in enumerate(prompt, start=1):

        if ch not in stoi:
            continue

        logits = gpt(
            stoi[ch],
            pos_id,
            keys,
            values
        )

    result = []

    pos_id = len(prompt) + 1

    for _ in range(max_tokens):

        if pos_id >= block_size:
            break

        probs = softmax([
            l / temperature
            for l in logits
        ])

        token_id = random.choices(
            range(vocab_size),
            weights=[p.data for p in probs]
        )[0]

        if token_id == BOS:
            break

        ch = uchars[token_id]

        result.append(ch)

        logits = gpt(
            token_id,
            pos_id,
            keys,
            values
        )

        pos_id += 1

        # Stop when a new user turn starts
        if "".join(result).endswith("\nUser:"):
            result = result[:-6]
            break

    return "".join(result)

print("\n================================")
print("          PrismGPT")
print("================================")
print("Type 'exit' to quit.\n")

history = ""

while True:

    user_input = input("You: ").strip()

    if user_input.lower() in ("exit", "quit"):
        print("GPT: Goodbye!")
        break

    if not user_input:
        continue

    history += "User: " + user_input + "\nAssistant:"

    response = generate(history)

    print("\nGPT:", response, "\n")

    history += response + "\n"

    # Keep recent conversation context
    history = history[-(block_size - 1):]
