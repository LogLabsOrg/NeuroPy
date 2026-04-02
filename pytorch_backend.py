"""
╔══════════════════════════════════════════════════════════════════════════╗
║               NeuroPy — Backend PyTorch                                 ║
║               Convierte nodos AST en código PyTorch real                ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import os
import json
import time
from typing import Dict, Any, List, Optional

# ── Intentar importar PyTorch ───────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset, random_split
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ── Intentar importar torchvision para datasets estándar ───────────────
try:
    import torchvision
    import torchvision.transforms as transforms
    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════
# MAPA DE CAPAS
# Traduce nombres NeuroPy → clases de PyTorch
# ═══════════════════════════════════════════════════════════════════════

def _get_activation(name: str):
    """Devuelve la función de activación de PyTorch por nombre."""
    activations = {
        "relu":       nn.ReLU(),
        "sigmoid":    nn.Sigmoid(),
        "tanh":       nn.Tanh(),
        "softmax":    nn.Softmax(dim=1),
        "gelu":       nn.GELU(),
        "silu":       nn.SiLU(),
        "leaky_relu": nn.LeakyReLU(),
        "elu":        nn.ELU(),
        "none":       nn.Identity(),
    }
    return activations.get(name.lower(), nn.ReLU())


def build_layer(layer_type: str, args: list, kwargs: dict, prev_out: int = 128) -> list:
    """
    Construye capas de PyTorch a partir de un LayerNode de NeuroPy.
    Retorna lista de módulos (puede incluir capa + activación).
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("[NeuroPy] PyTorch no está instalado. Ejecuta: pip install torch")

    t = layer_type.lower()
    modules = []
    activation_name = kwargs.get("activation", "none")

    # Dense / Linear
    if t in ("dense", "linear"):
        units = int(args[0]) if args else kwargs.get("units", 128)
        modules.append(nn.Linear(prev_out, int(units)))
        if activation_name != "none":
            modules.append(_get_activation(activation_name))

    # Conv2D
    elif t == "conv2d":
        filters  = int(args[0]) if args else kwargs.get("filters", 32)
        kernel   = int(kwargs.get("kernel",  3))
        stride   = int(kwargs.get("stride",  1))
        padding  = int(kwargs.get("padding", 1))
        in_ch    = kwargs.get("_in_channels", 1)
        modules.append(nn.Conv2d(in_ch, int(filters), kernel, stride, padding))
        if activation_name != "none":
            modules.append(_get_activation(activation_name))

    # Conv1D
    elif t == "conv1d":
        filters = int(args[0]) if args else 32
        kernel  = int(kwargs.get("kernel", 3))
        in_ch   = kwargs.get("_in_channels", 1)
        modules.append(nn.Conv1d(in_ch, int(filters), kernel, padding=1))
        if activation_name != "none":
            modules.append(_get_activation(activation_name))

    # MaxPool
    elif t in ("maxpool", "maxpool2d"):
        size = int(args[0]) if args else 2
        modules.append(nn.MaxPool2d(size))

    # AvgPool
    elif t in ("avgpool", "avgpool2d"):
        size = int(args[0]) if args else 2
        modules.append(nn.AvgPool2d(size))

    # GlobalAvgPool
    elif t == "globalavgpool":
        modules.append(nn.AdaptiveAvgPool2d(1))

    # Flatten
    elif t == "flatten":
        modules.append(nn.Flatten())

    # Dropout
    elif t == "dropout":
        rate = float(args[0]) if args else kwargs.get("rate", 0.5)
        modules.append(nn.Dropout(float(rate)))

    # BatchNorm
    elif t == "batchnorm":
        features = kwargs.get("features", prev_out)
        modules.append(nn.BatchNorm1d(int(features)))

    elif t == "batchnorm2d":
        features = kwargs.get("features", prev_out)
        modules.append(nn.BatchNorm2d(int(features)))

    # LSTM
    elif t == "lstm":
        units      = int(args[0]) if args else 128
        ret_seq    = kwargs.get("return_seq", False)
        input_size = kwargs.get("input_size", prev_out)
        modules.append(LSTMWrapper(input_size, int(units), bool(ret_seq)))

    # GRU
    elif t == "gru":
        units      = int(args[0]) if args else 128
        input_size = kwargs.get("input_size", prev_out)
        modules.append(GRUWrapper(input_size, int(units)))

    # Embedding
    elif t == "embedding":
        vocab = int(args[0]) if len(args) > 0 else 10000
        dim   = int(args[1]) if len(args) > 1 else 128
        modules.append(nn.Embedding(vocab, dim))

    # LayerNorm
    elif t == "layernorm":
        dim = int(args[0]) if args else prev_out
        modules.append(nn.LayerNorm(dim))

    # Residual placeholder
    elif t == "residual":
        modules.append(nn.Identity())

    else:
        print(f"[NeuroPy] Advertencia: capa '{layer_type}' no reconocida, se omite.")

    return modules


# ── Wrappers para capas recurrentes ────────────────────────────────────
class LSTMWrapper(nn.Module):
    def __init__(self, input_size, hidden_size, return_seq=False):
        super().__init__()
        self.lstm       = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.return_seq = return_seq

    def forward(self, x):
        out, _ = self.lstm(x)
        return out if self.return_seq else out[:, -1, :]


class GRUWrapper(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)

    def forward(self, x):
        out, _ = self.gru(x)
        return out[:, -1, :]


# ═══════════════════════════════════════════════════════════════════════
# BUILDER DE MODELO
# ═══════════════════════════════════════════════════════════════════════

class NeuroPyModel(nn.Module):
    """Red neuronal construida a partir de un CreateModelNode."""

    def __init__(self, layers_list: list):
        super().__init__()
        self.layers_seq = nn.Sequential(*layers_list)

    def forward(self, x):
        return self.layers_seq(x)


def build_model(create_node) -> "NeuroPyModel":
    """
    Recibe un CreateModelNode y devuelve un nn.Module de PyTorch.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("[NeuroPy] PyTorch no disponible.")

    all_modules = []
    prev_out    = 128  # tamaño de salida de la capa anterior (aproximado)

    for layer_node in create_node.layers:
        mods = build_layer(
            layer_node.layer_type,
            layer_node.args,
            layer_node.kwargs,
            prev_out=prev_out
        )
        all_modules.extend(mods)
        # Actualizar prev_out si es Dense
        if layer_node.layer_type.lower() in ("dense", "linear") and layer_node.args:
            prev_out = int(layer_node.args[0])

    model = NeuroPyModel(all_modules)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[NeuroPy] Modelo construido — {total_params:,} parámetros")
    return model


# ═══════════════════════════════════════════════════════════════════════
# MAPA DE PÉRDIDAS Y OPTIMIZADORES
# ═══════════════════════════════════════════════════════════════════════

LOSS_MAP = {
    "categorical_crossentropy": nn.CrossEntropyLoss,
    "binary_crossentropy":      nn.BCELoss,
    "mse":                      nn.MSELoss,
    "mae":                      nn.L1Loss,
    "huber":                    nn.HuberLoss,
}

def get_loss(name: str):
    cls = LOSS_MAP.get(name.lower())
    if cls is None:
        print(f"[NeuroPy] Pérdida '{name}' no encontrada, usando MSE.")
        cls = nn.MSELoss
    return cls()


def get_optimizer(model, name: str, kwargs: dict):
    lr = float(kwargs.get("lr", 0.001))
    name_l = name.lower()
    if name_l == "adam":
        return optim.Adam(model.parameters(), lr=lr,
                          betas=(float(kwargs.get("beta1", 0.9)),
                                 float(kwargs.get("beta2", 0.999))))
    elif name_l == "adamw":
        return optim.AdamW(model.parameters(), lr=lr,
                           weight_decay=float(kwargs.get("weight_decay", 0.01)))
    elif name_l == "sgd":
        return optim.SGD(model.parameters(), lr=lr,
                         momentum=float(kwargs.get("momentum", 0.9)))
    elif name_l == "rmsprop":
        return optim.RMSprop(model.parameters(), lr=lr)
    elif name_l == "adagrad":
        return optim.Adagrad(model.parameters(), lr=lr)
    else:
        print(f"[NeuroPy] Optimizador '{name}' no encontrado, usando Adam.")
        return optim.Adam(model.parameters(), lr=lr)


# ═══════════════════════════════════════════════════════════════════════
# ENTRENADOR PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════

def train_model(create_node, start_node, verbose: bool = True) -> Dict[str, Any]:
    """
    Ejecuta el entrenamiento completo de un modelo NeuroPy.

    Args:
        create_node: CreateModelNode con la configuración del modelo
        start_node:  StartBlockNode con save-data y opciones de inicio

    Returns:
        Diccionario con historial de métricas por época
    """
    if not TORCH_AVAILABLE:
        print("[NeuroPy] PyTorch no instalado. Simulando entrenamiento...")
        return _simulate_training(create_node)

    # ── Dispositivo ─────────────────────────────────────────────────
    device_str = create_node.device.lower()
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_str == "gpu":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[NeuroPy] Dispositivo: {device}")

    # ── Construir modelo ────────────────────────────────────────────
    model = build_model(create_node).to(device)

    # ── Pérdida y optimizador ───────────────────────────────────────
    criterion = get_loss(create_node.loss)
    optimizer = get_optimizer(model, create_node.optimizer, create_node.optimizer_kwargs)

    # ── Scheduler (reducir LR si no mejora) ────────────────────────
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=False)

    # ── Dataset (placeholder con datos aleatorios si no se provee) ──
    print("[NeuroPy] Preparando datos...")
    X = torch.randn(1000, 128)
    y = torch.randint(0, 10, (1000,))
    dataset = TensorDataset(X, y)
    train_size = int(0.8 * len(dataset))
    val_size   = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=create_node.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=create_node.batch_size)

    # ── Loop de entrenamiento ───────────────────────────────────────
    history = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}
    epochs  = create_node.epochs

    print(f"\n[NeuroPy] Iniciando entrenamiento — {epochs} épocas\n")
    bar_width = 30

    for epoch in range(1, epochs + 1):
        # — Train —
        model.train()
        train_loss  = 0.0
        train_correct = 0
        train_total   = 0
        t_start = time.time()

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss    = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss    += loss.item() * X_batch.size(0)
            preds          = outputs.argmax(dim=1)
            train_correct += (preds == y_batch).sum().item()
            train_total   += X_batch.size(0)

        train_loss /= train_total
        train_acc   = train_correct / train_total

        # — Validation —
        model.eval()
        val_loss    = 0.0
        val_correct = 0
        val_total   = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs  = model(X_batch)
                loss     = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                preds     = outputs.argmax(dim=1)
                val_correct += (preds == y_batch).sum().item()
                val_total   += X_batch.size(0)

        val_loss /= val_total
        val_acc   = val_correct / val_total
        scheduler.step(val_loss)

        # Guardar historial
        history["loss"].append(round(train_loss, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["accuracy"].append(round(train_acc, 4))
        history["val_accuracy"].append(round(val_acc, 4))

        # Barra de progreso
        elapsed = time.time() - t_start
        filled  = int(bar_width * epoch / epochs)
        bar     = "█" * filled + "░" * (bar_width - filled)
        print(
            f"  Época {epoch:>3}/{epochs}  [{bar}]  "
            f"loss={train_loss:.4f}  acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
            f"({elapsed:.1f}s)"
        )

    print(f"\n[NeuroPy] Entrenamiento completado.")

    # ── Guardar modelo ──────────────────────────────────────────────
    if create_node.save_path:
        _save_model(model, create_node, history)

    # ── Guardar datos de entrenamiento ──────────────────────────────
    if start_node and start_node.save_data_path:
        _save_training_data(history, start_node.save_data_path)

    return history


def _save_model(model, create_node, history):
    """Guarda el modelo en formato .pt y simula exportación .gguf."""
    path = create_node.save_path
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

    # Guardar pesos en .pt
    pt_path = path.replace(".gguf", ".pt").replace(".ggfu", ".pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_name":       create_node.model_name,
        "history":          history,
        "config": {
            "optimizer":  create_node.optimizer,
            "loss":       create_node.loss,
            "epochs":     create_node.epochs,
            "batch_size": create_node.batch_size,
        }
    }, pt_path)
    print(f"[NeuroPy] Modelo guardado → {pt_path}")

    # Nota sobre GGUF
    if ".gguf" in path or ".ggfu" in path:
        print(f"[NeuroPy] Nota: Exportación GGUF requiere llama.cpp.")
        print(f"          Para convertir: python convert.py {pt_path} --outfile {path}")


def _save_training_data(history: dict, path: str):
    """Guarda el historial de entrenamiento como JSON."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "history": history,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)
    print(f"[NeuroPy] Datos guardados → {path}")


def _simulate_training(create_node) -> dict:
    """Simula el entrenamiento cuando PyTorch no está disponible."""
    import random
    history = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}
    epochs  = create_node.epochs
    print(f"\n[NeuroPy] Simulando entrenamiento — {epochs} épocas (PyTorch no instalado)\n")
    loss = 2.5
    acc  = 0.1
    for epoch in range(1, epochs + 1):
        loss = max(0.05, loss * random.uniform(0.88, 0.96))
        acc  = min(0.99, acc  + random.uniform(0.02, 0.06))
        v_loss = loss + random.uniform(-0.05, 0.1)
        v_acc  = acc  - random.uniform(0.0,  0.05)
        history["loss"].append(round(loss,  4))
        history["val_loss"].append(round(v_loss, 4))
        history["accuracy"].append(round(acc,  4))
        history["val_accuracy"].append(round(v_acc, 4))
        print(f"  Época {epoch:>3}/{epochs}  loss={loss:.4f}  acc={acc:.4f}  val_loss={v_loss:.4f}  val_acc={v_acc:.4f}")
    print(f"\n[NeuroPy] Simulación completada.")
    return history
