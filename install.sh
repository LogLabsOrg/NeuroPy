#!/usr/bin/env bash
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                    NeuroPy Installer — install.sh                       ║
# ║                    LogLabs  ·  Versión 0.3.0                            ║
# ╠══════════════════════════════════════════════════════════════════════════╣
# ║  Uso:                                                                   ║
# ║    chmod +x install.sh                                                  ║
# ║    ./install.sh                                                          ║
# ║    ./install.sh --full       (incluye PyTorch + HuggingFace)            ║
# ║    ./install.sh --uninstall  (desinstala NeuroPy)                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

set -e

# ── Colores ────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

ok()   { echo -e "${GREEN}  ✓${RESET}  $1"; }
info() { echo -e "${CYAN}  →${RESET}  $1"; }
warn() { echo -e "${YELLOW}  !${RESET}  $1"; }
err()  { echo -e "${RED}  ✗${RESET}  $1"; }
hdr()  { echo -e "\n${BOLD}${CYAN}$1${RESET}"; echo "  $(echo "$1" | sed 's/./-/g')"; }

# ── Banner ─────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${CYAN}"
echo "  ███╗   ██╗███████╗██╗   ██╗██████╗  ██████╗ ██████╗ ██╗   ██╗"
echo "  ████╗  ██║██╔════╝██║   ██║██╔══██╗██╔═══██╗██╔══██╗╚██╗ ██╔╝"
echo "  ██╔██╗ ██║█████╗  ██║   ██║██████╔╝██║   ██║██████╔╝ ╚████╔╝ "
echo "  ██║╚██╗██║██╔══╝  ██║   ██║██╔══██╗██║   ██║██╔═══╝   ╚██╔╝  "
echo "  ██║ ╚████║███████╗╚██████╔╝██║  ██║╚██████╔╝██║        ██║   "
echo "  ╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚═╝        ╚═╝   "
echo -e "${RESET}"
echo -e "  ${BOLD}LogLabs  ·  Instalador v0.3.0${RESET}"
echo ""

# ── Argumentos ─────────────────────────────────────────────────────────────
FULL_INSTALL=false
UNINSTALL=false

for arg in "$@"; do
    case $arg in
        --full)       FULL_INSTALL=true ;;
        --uninstall)  UNINSTALL=true    ;;
        --help|-h)
            echo "  Uso: ./install.sh [opciones]"
            echo "  Opciones:"
            echo "    --full        Instalar PyTorch + HuggingFace + Matplotlib"
            echo "    --uninstall   Desinstalar NeuroPy"
            echo "    --help        Ver esta ayuda"
            exit 0 ;;
    esac
done

# ── Desinstalar ─────────────────────────────────────────────────────────────
if $UNINSTALL; then
    hdr "Desinstalando NeuroPy"

    INSTALL_DIR="$HOME/.neuropy"
    BIN_LINK="/usr/local/bin/neuropy"
    LOCAL_LINK="$HOME/.local/bin/neuropy"

    if [ -f "$BIN_LINK" ] || [ -L "$BIN_LINK" ]; then
        sudo rm -f "$BIN_LINK" 2>/dev/null && ok "Eliminado $BIN_LINK"
    fi
    if [ -f "$LOCAL_LINK" ] || [ -L "$LOCAL_LINK" ]; then
        rm -f "$LOCAL_LINK" && ok "Eliminado $LOCAL_LINK"
    fi
    if [ -d "$INSTALL_DIR" ]; then
        rm -rf "$INSTALL_DIR" && ok "Eliminado $INSTALL_DIR"
    fi

    # Limpiar shell configs
    for rc in "$HOME/.bashrc" "$HOME/.zshrc" "$HOME/.bash_profile" "$HOME/.profile"; do
        if [ -f "$rc" ]; then
            sed -i.bak '/# NeuroPy/d' "$rc" 2>/dev/null || true
            sed -i.bak '/\.neuropy/d' "$rc" 2>/dev/null || true
        fi
    done

    echo ""
    ok "NeuroPy desinstalado."
    exit 0
fi

# ── Verificar Python ────────────────────────────────────────────────────────
hdr "Verificando requisitos"

if ! command -v python3 &>/dev/null; then
    err "Python 3 no encontrado."
    echo "  Instálalo desde https://www.python.org/downloads/"
    exit 1
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || { [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 9 ]; }; then
    err "Se requiere Python 3.9+. Tienes $PYTHON_VERSION"
    exit 1
fi

ok "Python $PYTHON_VERSION encontrado"

if ! command -v pip3 &>/dev/null && ! python3 -m pip --version &>/dev/null; then
    err "pip no encontrado. Instala pip primero."
    exit 1
fi

ok "pip disponible"

# ── Directorio de instalación ───────────────────────────────────────────────
INSTALL_DIR="$HOME/.neuropy"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

hdr "Instalando NeuroPy en $INSTALL_DIR"

mkdir -p "$INSTALL_DIR"
mkdir -p "$INSTALL_DIR/backends"
mkdir -p "$INSTALL_DIR/libs"
mkdir -p "$INSTALL_DIR/examples"

# Copiar archivos del proyecto
info "Copiando archivos..."
cp "$SRC_DIR/neuropy_core.py"           "$INSTALL_DIR/"
cp "$SRC_DIR/backends/pytorch_backend.py" "$INSTALL_DIR/backends/" 2>/dev/null || warn "pytorch_backend.py no encontrado"
cp "$SRC_DIR/backends/hf_backend.py"     "$INSTALL_DIR/backends/" 2>/dev/null || warn "hf_backend.py no encontrado"
cp "$SRC_DIR/backends/viz_backend.py"    "$INSTALL_DIR/backends/" 2>/dev/null || warn "viz_backend.py no encontrado"

# Crear __init__.py para el paquete backends
touch "$INSTALL_DIR/backends/__init__.py"

ok "Archivos copiados"

# ── Crear ejecutable neuropy ────────────────────────────────────────────────
hdr "Creando comando 'neuropy'"

NEUROPY_BIN="$INSTALL_DIR/neuropy"

cat > "$NEUROPY_BIN" << SCRIPT
#!/usr/bin/env bash
# NeuroPy launcher — generado por install.sh
export NEUROPY_HOME="\$HOME/.neuropy"
exec python3 "\$NEUROPY_HOME/neuropy_core.py" "\$@"
SCRIPT

chmod +x "$NEUROPY_BIN"
ok "Ejecutable creado: $NEUROPY_BIN"

# ── Añadir al PATH ──────────────────────────────────────────────────────────
hdr "Configurando PATH"

# Intentar instalar en /usr/local/bin (requiere sudo)
install_system_wide() {
    if sudo ln -sf "$NEUROPY_BIN" /usr/local/bin/neuropy 2>/dev/null; then
        ok "Instalado en /usr/local/bin/neuropy (global)"
        return 0
    fi
    return 1
}

# Fallback: ~/.local/bin
install_user_local() {
    mkdir -p "$HOME/.local/bin"
    ln -sf "$NEUROPY_BIN" "$HOME/.local/bin/neuropy"
    ok "Instalado en ~/.local/bin/neuropy (usuario)"

    # Asegurarse de que ~/.local/bin esté en PATH
    LOCAL_BIN_IN_PATH=false
    echo "$PATH" | tr ':' '\n' | grep -q "$HOME/.local/bin" && LOCAL_BIN_IN_PATH=true

    if ! $LOCAL_BIN_IN_PATH; then
        PATH_LINE='export PATH="$HOME/.local/bin:$PATH" # NeuroPy'
        for rc in "$HOME/.bashrc" "$HOME/.zshrc" "$HOME/.bash_profile" "$HOME/.profile"; do
            if [ -f "$rc" ]; then
                echo "$PATH_LINE" >> "$rc"
                ok "PATH añadido a $rc"
            fi
        done
        warn "Reinicia la terminal o ejecuta: source ~/.bashrc"
    fi
}

if ! install_system_wide; then
    warn "No se pudo instalar globalmente (sin permisos sudo)."
    install_user_local
fi

# ── Instalar dependencias Python ────────────────────────────────────────────
hdr "Instalando dependencias Python"

PIP="python3 -m pip"

info "Instalando lark (parser)..."
$PIP install lark --quiet && ok "lark instalado" || err "Error instalando lark"

info "Instalando matplotlib..."
$PIP install matplotlib --quiet && ok "matplotlib instalado" || warn "matplotlib no instalado (opcional)"

if $FULL_INSTALL; then
    echo ""
    info "Instalación completa — descargando PyTorch..."
    warn "Esto puede tardar varios minutos..."
    echo ""

    # Detectar si hay GPU NVIDIA
    if command -v nvidia-smi &>/dev/null; then
        info "GPU NVIDIA detectada — instalando PyTorch con CUDA..."
        $PIP install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --quiet \
            && ok "PyTorch + CUDA instalado" || err "Error instalando PyTorch con CUDA"
    else
        info "Sin GPU — instalando PyTorch CPU..."
        $PIP install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet \
            && ok "PyTorch CPU instalado" || err "Error instalando PyTorch"
    fi

    info "Instalando transformers (HuggingFace)..."
    $PIP install transformers accelerate --quiet && ok "transformers instalado" || warn "transformers no instalado"

    info "Instalando llama-cpp-python (GGUF)..."
    $PIP install llama-cpp-python --quiet && ok "llama-cpp-python instalado" || warn "llama-cpp-python no instalado"

else
    echo ""
    warn "Instalación básica. Para PyTorch + HuggingFace ejecuta:"
    echo "    ./install.sh --full"
    echo ""
    info "O instala manualmente:"
    echo "    pip install torch transformers llama-cpp-python"
fi

# ── Verificar instalación ───────────────────────────────────────────────────
hdr "Verificando instalación"

if python3 -c "import lark" 2>/dev/null; then
    ok "lark importado correctamente"
else
    err "lark no se puede importar — algo salió mal"
fi

# ── Crear archivo de ejemplo ────────────────────────────────────────────────
EXAMPLE_FILE="$INSTALL_DIR/examples/hola_neuropy.npy"
cat > "$EXAMPLE_FILE" << 'EXAMPLE'
// hola_neuropy.npy — Tu primer programa NeuroPy
#NeuroPy <neuron>

const VERSION = "0.3.0"
print("Hola desde NeuroPy " + VERSION)

var x = 10
var y = 32
print("10 + 32 = " + str(x + y))
EXAMPLE

ok "Ejemplo creado: $EXAMPLE_FILE"

# ── Resumen final ───────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${GREEN}╔══════════════════════════════════════════════════════╗"
echo -e "║          NeuroPy instalado correctamente             ║"
echo -e "╚══════════════════════════════════════════════════════╝${RESET}"
echo ""
echo -e "  ${BOLD}Uso:${RESET}"
echo "    neuropy                      → Abrir REPL interactivo"
echo "    neuropy archivo.npy          → Ejecutar archivo"
echo "    neuropy --version            → Ver versión"
echo ""
echo -e "  ${BOLD}Prueba:${RESET}"
echo "    neuropy ~/.neuropy/examples/hola_neuropy.npy"
echo ""
echo -e "  ${BOLD}Extensiones .npy y .ny registradas.${RESET}"
echo ""

if ! command -v neuropy &>/dev/null; then
    warn "Si 'neuropy' no funciona aún, ejecuta:"
    echo "    source ~/.bashrc   (o reinicia la terminal)"
    echo ""
fi
