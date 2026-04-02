"""
╔══════════════════════════════════════════════════════════════════════════╗
║               NeuroPy — Backend HuggingFace                             ║
║               Carga y usa modelos preentrenados                          ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import os
from typing import Any, Dict, List, Optional

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    from transformers import AutoModelForSequenceClassification
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════
# MAPA DE TAREAS HuggingFace
# ═══════════════════════════════════════════════════════════════════════

TASK_MAP = {
    "classification":    "text-classification",
    "generation":        "text-generation",
    "summarization":     "summarization",
    "translation":       "translation",
    "ner":               "token-classification",
    "qa":                "question-answering",
    "embedding":         "feature-extraction",
    "image_class":       "image-classification",
    "object_detection":  "object-detection",
    "text_to_image":     "text-to-image",
    "fill_mask":         "fill-mask",
    "zero_shot":         "zero-shot-classification",
}


# ═══════════════════════════════════════════════════════════════════════
# CARGADOR PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════

class HuggingFaceLoader:
    """Carga y gestiona modelos desde HuggingFace Hub."""

    def __init__(self):
        self._loaded: Dict[str, Any] = {}

    def load(self, pretrained_node) -> Any:
        """
        Carga un modelo preentrenado a partir de un PretrainedNode.
        """
        name     = pretrained_node.name
        model_id = pretrained_node.model_id
        task_key = pretrained_node.task
        device   = pretrained_node.device

        if name in self._loaded:
            return self._loaded[name]

        hf_task = TASK_MAP.get(task_key.lower(), task_key)

        if not HF_AVAILABLE:
            print(f"[NeuroPy] HuggingFace no instalado. Simulando modelo '{model_id}'...")
            stub = _SimulatedModel(name, model_id, hf_task, pretrained_node.labels)
            self._loaded[name] = stub
            return stub

        print(f"[NeuroPy] Cargando '{model_id}' [{hf_task}]...")
        device_id = 0 if device in ("gpu", "auto") and _cuda_available() else -1

        try:
            pipe = pipeline(hf_task, model=model_id, device=device_id)
            wrapper = HFModelWrapper(name, model_id, pipe, pretrained_node.labels)
            self._loaded[name] = wrapper
            print(f"[NeuroPy] '{name}' listo.")
            return wrapper
        except Exception as e:
            print(f"[NeuroPy] Error cargando '{model_id}': {e}")
            stub = _SimulatedModel(name, model_id, hf_task, pretrained_node.labels)
            self._loaded[name] = stub
            return stub

    def get(self, name: str) -> Optional[Any]:
        return self._loaded.get(name)


class HFModelWrapper:
    """Envuelve un pipeline de HuggingFace con la interfaz de NeuroPy."""

    def __init__(self, name, model_id, pipe, labels):
        self.name     = name
        self.model_id = model_id
        self.pipe     = pipe
        self.labels   = labels

    def infer(self, text: str) -> str:
        try:
            result = self.pipe(text)
            if isinstance(result, list) and result:
                r = result[0]
                if "generated_text" in r:
                    return r["generated_text"]
                if "label" in r:
                    return f"{r['label']} ({r.get('score', 0):.2%})"
                if "summary_text" in r:
                    return r["summary_text"]
            return str(result)
        except Exception as e:
            return f"[Error de inferencia: {e}]"

    def __repr__(self):
        return f"<NeuroPy HFModel '{self.name}' — {self.model_id}>"


class _SimulatedModel:
    """Modelo simulado cuando HF no está disponible."""

    def __init__(self, name, model_id, task, labels):
        self.name     = name
        self.model_id = model_id
        self.task     = task
        self.labels   = labels

    def infer(self, text: str) -> str:
        import random
        if self.labels:
            label = random.choice(self.labels)
            score = random.uniform(0.7, 0.99)
            return f"{label} ({score:.2%}) [simulado]"
        return f"[Respuesta simulada de '{self.model_id}' para: '{text[:40]}...']"

    def __repr__(self):
        return f"<NeuroPy SimulatedModel '{self.name}'>"


# ═══════════════════════════════════════════════════════════════════════
# CARGADOR GGUF (llama.cpp)
# ═══════════════════════════════════════════════════════════════════════

class GGUFLoader:
    """Carga modelos .gguf con llama-cpp-python."""

    def __init__(self):
        self._loaded: Dict[str, Any] = {}

    def load(self, path: str, name: str = "model") -> Any:
        if name in self._loaded:
            return self._loaded[name]

        if not os.path.exists(path):
            print(f"[NeuroPy] Archivo no encontrado: {path}")
            stub = _SimulatedGGUF(name, path)
            self._loaded[name] = stub
            return stub

        if not LLAMA_AVAILABLE:
            print(f"[NeuroPy] llama-cpp-python no instalado.")
            print(f"          pip install llama-cpp-python")
            stub = _SimulatedGGUF(name, path)
            self._loaded[name] = stub
            return stub

        print(f"[NeuroPy] Cargando GGUF: {path}")
        try:
            model = Llama(model_path=path, n_ctx=2048, verbose=False)
            wrapper = GGUFWrapper(name, path, model)
            self._loaded[name] = wrapper
            print(f"[NeuroPy] GGUF '{name}' listo.")
            return wrapper
        except Exception as e:
            print(f"[NeuroPy] Error cargando GGUF: {e}")
            stub = _SimulatedGGUF(name, path)
            self._loaded[name] = stub
            return stub


class GGUFWrapper:
    def __init__(self, name, path, model):
        self.name  = name
        self.path  = path
        self.model = model

    def infer(self, prompt: str, max_tokens: int = 256) -> str:
        result = self.model(prompt, max_tokens=max_tokens, stop=["</s>", "\n\n"])
        return result["choices"][0]["text"].strip()

    def chat(self, user_msg: str) -> str:
        prompt = f"[INST] {user_msg} [/INST]"
        return self.infer(prompt)


class _SimulatedGGUF:
    def __init__(self, name, path):
        self.name = name
        self.path = path

    def infer(self, text: str, **kwargs) -> str:
        return f"[Respuesta simulada GGUF '{self.name}' para: '{text[:50]}']"

    def chat(self, text: str) -> str:
        return self.infer(text)


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# Instancia global
hf_loader   = HuggingFaceLoader()
gguf_loader = GGUFLoader()
