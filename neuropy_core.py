"""
╔══════════════════════════════════════════════════════════════════════════╗
║                        NeuroPy Language Core                            ║
║                   Lenguaje de IA  —  LogLabs                            ║
║                        Versión 0.3.0                                    ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Extensiones:   .npy  /  .ny                                            ║
║  Cabecera:      #NeuroPy <neuron>                                        ║
║  Librerías:     import %Nombre%.lib                                      ║
║  Comentarios:   // Este es un comentario                                 ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

from lark import Lark, Transformer, v_args, Token, Tree
from dataclasses import dataclass, field
from typing import Any, List, Optional, Dict
import sys, os, re

# ── Backends ───────────────────────────────────────────────────────────
_BACKEND_DIR = os.path.join(os.path.dirname(__file__), "backends")
sys.path.insert(0, _BACKEND_DIR)

try:
    from pytorch_backend import train_model, build_model, TORCH_AVAILABLE
except ImportError:
    TORCH_AVAILABLE = False
    def train_model(c, s, **kw): return {}
    def build_model(c): return None

try:
    from hf_backend import hf_loader, gguf_loader, HF_AVAILABLE
except ImportError:
    HF_AVAILABLE = False
    class _DummyLoader:
        def load(self, *a, **kw): return None
        def get(self, *a): return None
    hf_loader = gguf_loader = _DummyLoader()

try:
    from viz_backend import plot_metrics
except ImportError:
    def plot_metrics(m, h, **kw): print(f"[NeuroPy] plot {', '.join(m)}")


# ═══════════════════════════════════════════════════════════════════════
# GRAMÁTICA EBNF
# ═══════════════════════════════════════════════════════════════════════

NEUROPY_GRAMMAR = r"""
    start: header import_stmt* statement*

    header: "#NeuroPy" "<" NAME ">" NEWLINE

    import_stmt: "import" IMPORT_PATH NEWLINE

    statement: var_decl
             | const_decl
             | create_stmt
             | start_block
             | window_decl
             | window_prop
             | connect_stmt
             | ofm_block
             | prompt_stmt
             | input_stmt
             | print_stmt
             | return_stmt
             | if_stmt
             | while_stmt
             | for_stmt
             | func_decl
             | plot_stmt
             | comment

    var_decl   : "var"   NAME "=" expr NEWLINE
    const_decl : "const" NAME "=" expr NEWLINE

    // ── CREATE ─────────────────────────────────────────────────────
    create_stmt: "create" create_type "{" NEWLINE create_body "}" NEWLINE?
               | "create" "window" NEWLINE?

    create_type: NAME

    create_body: create_item+

    create_item: "name"      "model"    "(" STRING ")"           NEWLINE
               | "optimizer" NAME       call_args?               NEWLINE
               | "loss"      NAME                                NEWLINE
               | "epochs"    NUMBER                              NEWLINE
               | "batch"     NUMBER                              NEWLINE
               | "metric"    NAME ("," NAME)*                    NEWLINE
               | "device"    NAME                                NEWLINE
               | "on"        event_name "->" action              NEWLINE
               | "save-model" STRING                             NEWLINE
               | "layer"     NAME call_args?                     NEWLINE
               | "input"     shape                               NEWLINE
               | "output"    shape                               NEWLINE

    // ── START BLOCK ────────────────────────────────────────────────
    start_block: "start" "create" NAME "(" NEWLINE start_body ")" "end" NEWLINE?

    start_body: start_item+

    start_item: "save-data" STRING                               NEWLINE
              | print_stmt
              | var_decl
              | comment

    // ── WINDOW ─────────────────────────────────────────────────────
    window_decl : "window" "name" "(" STRING ")"                 NEWLINE
    window_prop : "ancho"  NUMBER                                 NEWLINE
                | "largo"  NUMBER                                 NEWLINE
                | "into"   "(" NEWLINE into_body ")"             NEWLINE?

    into_body: into_item+

    into_item: "create" "window" INSTANCE_ID                     NEWLINE
             | "window" "type"   "(" NAME ")"                    NEWLINE
             | "chat"   "name"   "(" STRING ")"                  NEWLINE
             | input_for_stmt
             | start_model_stmt
             | input_id_stmt
             | connect_stmt
             | "messagebox" "(" USER_INPUT ")"                   NEWLINE
             | ofm_block
             | print_stmt
             | comment

    // ── INPUTS Y CONEXIONES ────────────────────────────────────────
    input_for_stmt  : "input" "for" NAME "(" USER_INPUT ")"      NEWLINE
    input_id_stmt   : "input" STRING "(" MODEL_ID ")"            NEWLINE
    start_model_stmt: "Start" IMPORT_PATH                        NEWLINE

    connect_stmt: "connect" "to" connect_target                  NEWLINE
                | "connect" "(" NAME "+" connect_target          NEWLINE

    connect_target: "window" "(" INSTANCE_ID ")"
                  | "Console"
                  | NAME

    // ── OFM ────────────────────────────────────────────────────────
    ofm_block: "ofm" IMPORT_PATH "(" NEWLINE ofm_body ")" NEWLINE?

    ofm_body: ofm_item+

    ofm_item: "send"     "(" "imf" STRING ")"                    NEWLINE
            | "response" "(" MODEL_ID ")"                        NEWLINE
            | "end" "ofm"                                        NEWLINE
            | comment

    // ── PROMPT ─────────────────────────────────────────────────────
    prompt_stmt: "prompt" IMPORT_PATH "(" STRING ")"             NEWLINE

    // ── EVENTOS ────────────────────────────────────────────────────
    event_name: NAME ("_" NAME)*
    action    : "plot" NAME ("," NAME)*
              | "save" STRING
              | "log"  expr

    // ── FUNCIONES ──────────────────────────────────────────────────
    func_decl : "func" NAME "(" param_list? ")" "{" NEWLINE statement* "}" NEWLINE?
    param_list: param ("," param)*
    param     : NAME ("=" expr)?
    return_stmt: "return" expr? NEWLINE

    // ── CONTROL DE FLUJO ───────────────────────────────────────────
    if_stmt   : "if" expr "{" NEWLINE statement* "}"
                ("else" "if" expr "{" NEWLINE statement* "}")*
                ("else" "{" NEWLINE statement* "}")? NEWLINE?

    while_stmt: "while" expr "{" NEWLINE statement* "}" NEWLINE?
    for_stmt  : "for" NAME "in" expr "{" NEWLINE statement* "}" NEWLINE?

    // ── SALIDA ─────────────────────────────────────────────────────
    print_stmt: "print" "(" expr_list ")"                        NEWLINE
    plot_stmt : "plot"  NAME ("," NAME)*                         NEWLINE

    input_stmt: input_for_stmt | input_id_stmt

    // ── EXPRESIONES ────────────────────────────────────────────────
    expr_list : expr ("+" expr)*

    expr: comparison

    comparison : addition (comp_op addition)*
    comp_op    : "==" | "!=" | ">=" | "<=" | ">" | "<" | "and" | "or"

    addition       : multiplication (("+" | "-") multiplication)*
    multiplication : unary (("*" | "/" | "%") unary)*
    unary          : "-" unary | "not" unary | power
    power          : atom ("**" unary)?

    atom: NUMBER  -> number
        | STRING  -> string
        | BOOL    -> bool_val
        | NAME    -> name_ref
        | "null"  -> null_val
        | "(" expr ")"
        | list_expr
        | dict_expr

    list_expr: "[" (expr ("," expr)*)? "]"
    dict_expr: "{" (kv_pair ("," kv_pair)*)? "}"
    kv_pair  : (STRING | NAME) ":" expr

    call_args    : "(" (call_arg ("," call_arg)*)? ")"
    call_arg     : NAME "=" expr  -> named_arg
                 | expr           -> pos_arg

    shape: "[" dim ("," dim)* "]"
    dim  : NUMBER | "None" | NAME

    // ── TERMINALES ─────────────────────────────────────────────────
    NAME        : /[a-zA-Z_][a-zA-Z0-9_]*/
    NUMBER      : /\d+(\.\d+)?/
    STRING      : /\"[^\"]*\"|\'[^\']*\'/
    BOOL        : "true" | "false"
    NEWLINE     : /\n/
    IMPORT_PATH : /[%]?[\w\/\-\.]+[%]?\.(?:lib|gguf|ggfu)/
    INSTANCE_ID : /\*[0-9]+/
    USER_INPUT  : "&"
    MODEL_ID    : "@"

    comment: /\/\/[^\n]*/

    %ignore /[ \t]+/
    %ignore /\r/
"""


# ═══════════════════════════════════════════════════════════════════════
# NODOS DEL AST
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Node: pass

@dataclass
class ProgramNode(Node):
    engine: str
    imports: List[str]
    statements: List[Node]

@dataclass
class VarDeclNode(Node):
    name: str
    value: Node
    mutable: bool = True

@dataclass
class NameRefNode(Node):
    name: str

@dataclass
class AssignNode(Node):
    name: str; value: Node; op: str = "="

@dataclass
class NumberNode(Node):  value: float
@dataclass
class StringNode(Node):  value: str
@dataclass
class BoolNode(Node):    value: bool
@dataclass
class NullNode(Node):    pass
@dataclass
class ListNode(Node):    elements: List[Node]
@dataclass
class DictNode(Node):    pairs: List[tuple]

@dataclass
class BinOpNode(Node):
    left: Node; op: str; right: Node

@dataclass
class UnaryOpNode(Node):
    op: str; operand: Node

@dataclass
class ShapeNode(Node):
    dims: List[Any]

@dataclass
class LayerNode(Node):
    layer_type: str
    args: List[Any]
    kwargs: Dict[str, Any]

@dataclass
class CreateModelNode(Node):
    create_type: str
    model_name: str
    optimizer: str
    optimizer_kwargs: Dict[str, Any]
    loss: str
    epochs: int
    batch_size: int
    metrics: List[str]
    device: str
    save_path: str
    callbacks: List[tuple]
    layers: List[LayerNode]
    input_shape: Optional[ShapeNode]
    output_shape: Optional[ShapeNode]

@dataclass
class StartBlockNode(Node):
    target: str
    save_data_path: Optional[str]
    body: List[Node]

@dataclass
class WindowDeclNode(Node):
    app_name: str
    width: Optional[int] = None
    height: Optional[int] = None
    children: List[Node] = field(default_factory=list)

@dataclass
class OFMBlockNode(Node):
    lib_path: str
    send_input_for: str
    response_id: str

@dataclass
class PromptNode(Node):
    lib_path: str; prompt_text: str

@dataclass
class InputForNode(Node):  model_name: str
@dataclass
class InputIdNode(Node):   model_name: str
@dataclass
class StartModelNode(Node): lib_path: str
@dataclass
class ConnectNode(Node):
    source: str; target: str; instance_id: Optional[str] = None

@dataclass
class IfNode(Node):
    condition: Node
    then_body: List[Node]
    else_ifs: List[tuple]
    else_body: Optional[List[Node]]

@dataclass
class WhileNode(Node):
    condition: Node; body: List[Node]

@dataclass
class ForNode(Node):
    var_name: str; iterable: Node; body: List[Node]

@dataclass
class FuncDeclNode(Node):
    name: str; params: List[tuple]; body: List[Node]

@dataclass
class ReturnNode(Node):
    value: Optional[Node]

@dataclass
class PrintNode(Node):
    parts: List[Node]

@dataclass
class PlotNode(Node):
    metrics: List[str]

@dataclass
class PretrainedNode(Node):
    name: str; hub: str; model_id: str
    task: str; labels: List[str]
    finetune: bool; device: str


# ═══════════════════════════════════════════════════════════════════════
# TRANSFORMER
# ═══════════════════════════════════════════════════════════════════════

@v_args(inline=True)
class NeuroPyTransformer(Transformer):

    def start(self, header, *rest):
        imports = [r for r in rest if isinstance(r, str)]
        stmts   = [r for r in rest if isinstance(r, Node)]
        return ProgramNode(engine=header, imports=imports, statements=stmts)

    def header(self, _kw, engine, _nl=None):
        return str(engine)

    def import_stmt(self, path, _nl=None):
        return str(path)

    def var_decl(self, name, value, _nl=None):
        return VarDeclNode(name=str(name), value=value, mutable=True)

    def const_decl(self, name, value, _nl=None):
        return VarDeclNode(name=str(name), value=value, mutable=False)

    def number(self, t):
        v = float(str(t)); return NumberNode(value=int(v) if v == int(v) else v)
    def string(self, t):  return StringNode(value=str(t)[1:-1])
    def bool_val(self, t):return BoolNode(value=(str(t) == "true"))
    def null_val(self):   return NullNode()
    def name_ref(self, t):return NameRefNode(name=str(t))
    def list_expr(self, *e): return ListNode(elements=list(e))
    def dict_expr(self, *p): return DictNode(pairs=list(p))
    def kv_pair(self, k, v): return (str(k).strip('"\''), v)

    def _binop(self, left, *rest):
        result = left
        i = 0
        while i < len(rest):
            op = str(rest[i]) if isinstance(rest[i], Token) else rest[i].data
            result = BinOpNode(left=result, op=op, right=rest[i+1])
            i += 2
        return result

    def comparison(self, left, *rest):    return self._binop(left, *rest)
    def addition(self, left, *rest):      return self._binop(left, *rest)
    def multiplication(self, left, *rest):return self._binop(left, *rest)
    def unary(self, op, operand):         return UnaryOpNode(op=str(op), operand=operand)
    def power(self, base, *rest):
        return BinOpNode(left=base, op="**", right=rest[0]) if rest else base

    def shape(self, *dims):
        return ShapeNode(dims=[
            None if str(d)=="None" else
            int(float(str(d))) if str(d).replace(".","").isdigit() else str(d)
            for d in dims
        ])

    def create_type(self, *tokens):
        return "_".join(str(t) for t in tokens if isinstance(t, Token))

    def create_stmt(self, *args):
        # "create window" sin cuerpo
        if not args or (len(args)==1 and isinstance(args[0], Token)):
            return WindowDeclNode(app_name="")

        ct   = str(args[0]) if not isinstance(args[0], Tree) else "Model"
        body = args[1] if len(args) > 1 else None

        model_name = ""; optimizer = "Adam"; opt_kw = {}
        loss = "categorical_crossentropy"; epochs = 10; batch = 32
        metrics = []; device = "auto"; save_path = ""; callbacks = []
        layers = []; input_shape = None; output_shape = None

        if body and hasattr(body, "children"):
            for item in body.children:
                if not isinstance(item, Tree): continue
                if item.data != "create_item": continue
                ch = item.children
                if not ch: continue
                first = str(ch[0])
                if first == "name":      model_name = str(ch[2]).strip('"\'')
                elif first == "optimizer": optimizer = str(ch[1])
                elif first == "loss":    loss       = str(ch[1])
                elif first == "epochs":  epochs     = int(float(str(ch[1])))
                elif first == "batch":   batch      = int(float(str(ch[1])))
                elif first == "metric":  metrics    = [str(c) for c in ch[1:] if isinstance(c,Token) and str(c)!=","]
                elif first == "device":  device     = str(ch[1])
                elif first == "save-model": save_path = str(ch[1]).strip('"\'')
                elif first == "layer":
                    lt = str(ch[1])
                    layers.append(LayerNode(layer_type=lt, args=[], kwargs={}))
                elif first == "input":
                    if isinstance(ch[1], ShapeNode): input_shape = ch[1]
                elif first == "output":
                    if isinstance(ch[1], ShapeNode): output_shape = ch[1]

        return CreateModelNode(
            create_type=ct, model_name=model_name,
            optimizer=optimizer, optimizer_kwargs=opt_kw,
            loss=loss, epochs=epochs, batch_size=batch,
            metrics=metrics, device=device, save_path=save_path,
            callbacks=callbacks, layers=layers,
            input_shape=input_shape, output_shape=output_shape
        )

    def start_block(self, target, _nl, body, *_):
        save_path = None; stmts = []
        if hasattr(body, "children"):
            for item in body.children:
                if isinstance(item, Tree) and item.data == "start_item":
                    ch = item.children
                    if ch and str(ch[0]) == "save-data":
                        save_path = str(ch[1]).strip('"\'')
                elif isinstance(item, Node):
                    stmts.append(item)
        return StartBlockNode(target=str(target), save_data_path=save_path, body=stmts)

    def window_decl(self, name_str, _nl=None):
        return WindowDeclNode(app_name=str(name_str).strip('"\''))

    def window_prop(self, *args): return None

    def ofm_block(self, lib_path, _nl, body, *_):
        send_for = ""; resp_id = ""
        if hasattr(body, "children"):
            for item in body.children:
                if isinstance(item, Tree):
                    ch = item.children
                    if ch:
                        if str(ch[0]) == "send":
                            send_for = str(ch[-1]).strip('"\'')
                        elif str(ch[0]) == "response":
                            resp_id = str(ch[-1])
        return OFMBlockNode(lib_path=str(lib_path), send_input_for=send_for, response_id=resp_id)

    def prompt_stmt(self, lib_path, text, _nl=None):
        return PromptNode(lib_path=str(lib_path), prompt_text=str(text).strip('"\''))

    def input_for_stmt(self, model_name, _u, _nl=None):
        return InputForNode(model_name=str(model_name))

    def input_id_stmt(self, model_name, _m, _nl=None):
        return InputIdNode(model_name=str(model_name).strip('"\''))

    def start_model_stmt(self, lib_path, _nl=None):
        return StartModelNode(lib_path=str(lib_path))

    def connect_stmt(self, *args):
        return ConnectNode(source="self", target=str(args[-1]) if args else "")

    def if_stmt(self, condition, *rest):
        stmts = [r for r in rest if isinstance(r, Node)]
        return IfNode(condition=condition, then_body=stmts, else_ifs=[], else_body=None)

    def while_stmt(self, condition, *stmts):
        return WhileNode(condition=condition, body=[s for s in stmts if isinstance(s, Node)])

    def for_stmt(self, var_name, iterable, *stmts):
        return ForNode(var_name=str(var_name), iterable=iterable,
                       body=[s for s in stmts if isinstance(s, Node)])

    def func_decl(self, name, *args):
        body = [a for a in args if isinstance(a, Node)]
        return FuncDeclNode(name=str(name), params=[], body=body)

    def return_stmt(self, *args):
        value = args[0] if args and isinstance(args[0], Node) else None
        return ReturnNode(value=value)

    def expr_list(self, *parts):
        return list(parts)

    def print_stmt(self, parts_or_expr, _nl=None):
        if isinstance(parts_or_expr, list):
            return PrintNode(parts=parts_or_expr)
        return PrintNode(parts=[parts_or_expr])

    def plot_stmt(self, *args):
        metrics = [str(a) for a in args if isinstance(a, Token) and str(a) != ","]
        return PlotNode(metrics=metrics)

    def comment(self, *_): return None
    def input_stmt(self, child): return child


# ═══════════════════════════════════════════════════════════════════════
# ENVIRONMENT
# ═══════════════════════════════════════════════════════════════════════

class Environment:
    def __init__(self, parent=None):
        self._vars: Dict[str, Any] = {}
        self._consts: set = set()
        self.parent = parent

    def define(self, name, value, is_const=False):
        self._vars[name] = value
        if is_const: self._consts.add(name)

    def assign(self, name, value):
        if name in self._vars:
            if name in self._consts:
                raise RuntimeError(f"[NeuroPy] '{name}' es constante.")
            self._vars[name] = value
        elif self.parent:
            self.parent.assign(name, value)
        else:
            raise RuntimeError(f"[NeuroPy] Variable '{name}' no definida.")

    def get(self, name):
        if name in self._vars: return self._vars[name]
        if self.parent: return self.parent.get(name)
        raise RuntimeError(f"[NeuroPy] Variable '{name}' no definida.")

    def child(self): return Environment(parent=self)


# ═══════════════════════════════════════════════════════════════════════
# RUNTIME PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════

class NeuroPyRuntime:
    def __init__(self):
        self.global_env  = Environment()
        self._models:    Dict[str, CreateModelNode] = {}
        self._pretrained:Dict[str, Any]             = {}
        self._functions: Dict[str, FuncDeclNode]    = {}
        self._libs:      Dict[str, Any]             = {}
        self._history:   Dict[str, list]            = {}
        self._prompt:    str                        = "NeuroPy > "
        self._active_model_id: Optional[str]        = None
        self._register_builtins()

    def _register_builtins(self):
        for name, fn in {
            "abs": abs, "max": max, "min": min,
            "round": round, "len": len, "str": str,
            "int": int, "float": float, "range": range,
            "type": lambda x: type(x).__name__,
        }.items():
            self.global_env.define(name, fn)

    def eval(self, node, env=None):
        if node is None: return None
        if env is None: env = self.global_env
        handler = getattr(self, f"_eval_{type(node).__name__}", None)
        return handler(node, env) if handler else None

    # ── Programa ────────────────────────────────────────────────────
    def _eval_ProgramNode(self, node: ProgramNode, env):
        print(f"[NeuroPy] Motor: <{node.engine}>")
        for lib in node.imports:
            self._load_lib(lib)
        result = None
        for stmt in node.statements:
            if stmt is not None:
                result = self.eval(stmt, env)
        return result

    def _load_lib(self, path: str):
        ext = os.path.splitext(path)[1].lower()
        print(f"[NeuroPy] import {path}")
        if ext in (".gguf", ".ggfu"):
            real_path = path.replace("%", "")
            model = gguf_loader.load(real_path, name=os.path.basename(real_path))
            self._libs[path] = model
        else:
            self._libs[path] = {"loaded": True, "path": path}

    # ── Literales ───────────────────────────────────────────────────
    def _eval_NumberNode(self, n, e): return n.value
    def _eval_StringNode(self, n, e): return n.value
    def _eval_BoolNode(self,   n, e): return n.value
    def _eval_NullNode(self,   n, e): return None
    def _eval_ListNode(self,   n, e): return [self.eval(x, e) for x in n.elements]
    def _eval_DictNode(self,   n, e): return {k: self.eval(v, e) for k, v in n.pairs}

    def _eval_NameRefNode(self, node, env):
        try: return env.get(node.name)
        except: return node.name

    def _eval_VarDeclNode(self, node, env):
        v = self.eval(node.value, env)
        env.define(node.name, v, is_const=not node.mutable)
        return v

    def _eval_AssignNode(self, node, env):
        v = self.eval(node.value, env)
        env.assign(node.name, v); return v

    def _eval_BinOpNode(self, node, env):
        l = self.eval(node.left,  env)
        r = self.eval(node.right, env)
        return {
            "+": lambda a,b: str(a)+str(b) if isinstance(a,str) or isinstance(b,str) else a+b,
            "-": lambda a,b: a-b, "*": lambda a,b: a*b, "/": lambda a,b: a/b,
            "%": lambda a,b: a%b, "**": lambda a,b: a**b,
            "==": lambda a,b: a==b, "!=": lambda a,b: a!=b,
            ">": lambda a,b: a>b,  "<": lambda a,b: a<b,
            ">=": lambda a,b: a>=b,"<=": lambda a,b: a<=b,
            "and": lambda a,b: a and b,"or": lambda a,b: a or b,
        }.get(node.op, lambda a,b: None)(l, r)

    def _eval_UnaryOpNode(self, node, env):
        v = self.eval(node.operand, env)
        return -v if node.op == "-" else not v

    # ── CREATE ──────────────────────────────────────────────────────
    def _eval_CreateModelNode(self, node: CreateModelNode, env):
        print(f"\n[NeuroPy] create {node.create_type}")
        if node.model_name:
            print(f"          name model : \"{node.model_name}\"")
        print(f"          optimizer  : {node.optimizer}")
        print(f"          loss       : {node.loss}")
        print(f"          epochs     : {node.epochs}")
        print(f"          batch      : {node.batch_size}")
        print(f"          device     : {node.device}")
        if node.save_path:
            print(f"          save-model : {node.save_path}")
        self._models[node.create_type] = node
        env.define(node.create_type, node)
        return node

    # ── START ───────────────────────────────────────────────────────
    def _eval_StartBlockNode(self, node: StartBlockNode, env):
        model_node = self._models.get(node.target)
        if not model_node:
            raise RuntimeError(f"[NeuroPy] 'create {node.target}' no fue definido.")

        print(f"\n[NeuroPy] start create {node.target}")
        if node.save_data_path:
            print(f"          save-data  : {node.save_data_path}")

        # Entrenar con PyTorch
        history = train_model(model_node, node, verbose=True)
        self._history[node.target] = history

        # Ejecutar cuerpo del start
        for stmt in node.body:
            self.eval(stmt, env)

        print(f"\n[NeuroPy] end — {node.target} finalizado.\n")
        return {"status": "done", "model": node.target, "history": history}

    # ── WINDOW ──────────────────────────────────────────────────────
    def _eval_WindowDeclNode(self, node: WindowDeclNode, env):
        if node.app_name:
            print(f"[NeuroPy] window name(\"{node.app_name}\")")
        return node

    # ── OFM ─────────────────────────────────────────────────────────
    def _eval_OFMBlockNode(self, node: OFMBlockNode, env):
        # Intentar usar modelo real si está cargado
        model = self._libs.get(node.lib_path) or hf_loader.get(node.send_input_for)
        if model and hasattr(model, "chat"):
            print(f"[NeuroPy] ofm {node.lib_path} activo.")
        else:
            print(f"[NeuroPy] ofm {node.lib_path} configurado.")
        return node

    # ── PROMPT ──────────────────────────────────────────────────────
    def _eval_PromptNode(self, node: PromptNode, env):
        self._prompt = node.prompt_text
        print(f"[NeuroPy] prompt → \"{self._prompt}\"")
        return node

    # ── INPUTS / CONNECT ────────────────────────────────────────────
    def _eval_InputForNode(self, n, e):
        print(f"[NeuroPy] input for {n.model_name} (&)"); return n
    def _eval_InputIdNode(self, n, e):
        self._active_model_id = n.model_name
        print(f"[NeuroPy] input \"{n.model_name}\"(@)"); return n
    def _eval_StartModelNode(self, n, e):
        print(f"[NeuroPy] Start {n.lib_path}"); return n
    def _eval_ConnectNode(self, n, e):
        print(f"[NeuroPy] connect → {n.target}"); return n

    # ── CONTROL DE FLUJO ────────────────────────────────────────────
    def _eval_IfNode(self, node, env):
        if self.eval(node.condition, env):
            child = env.child()
            for s in node.then_body: self.eval(s, child)
        else:
            for cond, body in node.else_ifs:
                if self.eval(cond, env):
                    child = env.child()
                    for s in body: self.eval(s, child); return
            if node.else_body:
                child = env.child()
                for s in node.else_body: self.eval(s, child)

    def _eval_WhileNode(self, node, env):
        while self.eval(node.condition, env):
            child = env.child()
            for s in node.body: self.eval(s, child)

    def _eval_ForNode(self, node, env):
        for item in (self.eval(node.iterable, env) or []):
            child = env.child()
            child.define(node.var_name, item)
            for s in node.body: self.eval(s, child)

    # ── FUNCIONES ───────────────────────────────────────────────────
    def _eval_FuncDeclNode(self, node, env):
        self._functions[node.name] = node
        env.define(node.name, node); return node

    def _eval_ReturnNode(self, node, env):
        raise _ReturnException(self.eval(node.value, env) if node.value else None)

    # ── SALIDA ──────────────────────────────────────────────────────
    def _eval_PrintNode(self, node: PrintNode, env):
        parts = [str(self.eval(p, env)) for p in node.parts]
        out = "".join(parts)
        print(out); return out

    def _eval_PlotNode(self, node: PlotNode, env):
        last_history = None
        for h in self._history.values():
            last_history = h
        if last_history:
            plot_metrics(node.metrics, last_history)
        else:
            print(f"[NeuroPy] plot {', '.join(node.metrics)} — sin datos aún")


class _ReturnException(Exception):
    def __init__(self, value): self.value = value


# ═══════════════════════════════════════════════════════════════════════
# PRE-PROCESADOR
# ═══════════════════════════════════════════════════════════════════════

def preprocess(source: str) -> str:
    lines = []
    for line in source.splitlines():
        idx = line.find("//")
        if idx != -1:
            # No cortar si // está dentro de una string
            in_str = False; ch_prev = ""
            for i, ch in enumerate(line):
                if ch in ('"', "'") and ch_prev != "\\":
                    in_str = not in_str
                if not in_str and line[i:i+2] == "//":
                    line = line[:i]; break
                ch_prev = ch
        lines.append(line)
    return "\n".join(lines) + "\n"


def validate_header(source: str) -> bool:
    first = source.strip().splitlines()[0] if source.strip() else ""
    if not re.match(r"^#NeuroPy\s+<\w+>", first):
        print(
            "\n╔══ NeuroPy Error ══════════════════════════════╗\n"
            "║  El archivo debe comenzar con:                ║\n"
            "║                                               ║\n"
            "║      #NeuroPy <neuron>                        ║\n"
            "║                                               ║\n"
            "║  Sin esta cabecera el motor no arranca.       ║\n"
            "╚═══════════════════════════════════════════════╝\n"
        )
        return False
    return True


# ═══════════════════════════════════════════════════════════════════════
# INTERFAZ PÚBLICA
# ═══════════════════════════════════════════════════════════════════════

class NeuroPyInterpreter:
    def __init__(self):
        self.parser      = Lark(NEUROPY_GRAMMAR, parser="earley", ambiguity="resolve")
        self.transformer = NeuroPyTransformer()
        self.runtime     = NeuroPyRuntime()

    def parse(self, source: str):
        processed = preprocess(source)
        tree = self.parser.parse(processed)
        return self.transformer.transform(tree)

    def run(self, source: str):
        if not validate_header(source): return None
        ast = self.parse(source)
        return self.runtime.eval(ast)

    def run_file(self, path: str):
        ext = os.path.splitext(path)[1].lower()
        if ext not in (".npy", ".ny"):
            print(f"[NeuroPy] Advertencia: se esperaba .npy o .ny — recibido '{ext}'")
        with open(path, "r", encoding="utf-8") as f:
            source = f.read()
        return self.run(source)


# ═══════════════════════════════════════════════════════════════════════
# CLI — punto de entrada de `neuropy`
# ═══════════════════════════════════════════════════════════════════════

BANNER = """
╔═══════════════════════════════════════════════════════╗
║          NeuroPy 0.3.0  —  LogLabs                    ║
║          Lenguaje de IA con motor <neuron>            ║
╠═══════════════════════════════════════════════════════╣
║  neuropy archivo.npy      Ejecutar archivo            ║
║  neuropy                  Iniciar REPL                ║
║  neuropy --version        Ver versión                 ║
║  neuropy --help           Ver ayuda                   ║
╚═══════════════════════════════════════════════════════╝
"""

def main():
    interp = NeuroPyInterpreter()
    args   = sys.argv[1:]

    if "--version" in args or "-v" in args:
        print("NeuroPy 0.3.0  —  LogLabs"); return

    if "--help" in args or "-h" in args:
        print(BANNER); return

    if args:
        path = args[0]
        if not os.path.exists(path):
            print(f"[NeuroPy] Archivo no encontrado: {path}"); sys.exit(1)
        interp.run_file(path)
        return

    # ── REPL ────────────────────────────────────────────────────────
    print(BANNER)
    buffer = []
    while True:
        try:
            prompt_sym = "... " if buffer else ">>> "
            line = input(prompt_sym)
            if line.lower().strip() in ("exit", "salir", "quit"):
                print("[NeuroPy] Hasta pronto."); break
            buffer.append(line)
            if line.strip() == "" and buffer:
                code = "\n".join(b for b in buffer if b.strip())
                buffer.clear()
                if code.strip():
                    try:
                        result = interp.run(code)
                        if result is not None:
                            print(f"  => {result}")
                    except Exception as e:
                        print(f"[NeuroPy] Error: {e}")
        except (KeyboardInterrupt, EOFError):
            print(); break


if __name__ == "__main__":
    main()
