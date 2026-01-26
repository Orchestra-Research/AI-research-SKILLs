#!/bin/bash
# =============================================================================
# AI Research Skills Installation Script for Claude Code
# =============================================================================
# This script installs the AI Research Engineering Skills Library for use
# with Claude Code on any machine.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/hypnopump/AI-research-SKILLs/main/install.sh | bash
#
#   Or clone and run locally:
#   git clone https://github.com/hypnopump/AI-research-SKILLs.git
#   cd AI-research-SKILLs && ./install.sh
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
REPO_URL="${REPO_URL:-https://github.com/hypnopump/AI-research-SKILLs.git}"
INSTALL_DIR="${INSTALL_DIR:-$HOME/.claude/skills/ai-research-skills}"
SKILLS_CONFIG_DIR="${SKILLS_CONFIG_DIR:-$HOME/.claude}"

print_banner() {
    echo -e "${BLUE}"
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║         AI Research Engineering Skills Library                ║"
    echo "║                   Installation Script                         ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_dependencies() {
    log_info "Checking dependencies..."

    # Check for git
    if ! command -v git &> /dev/null; then
        log_error "git is required but not installed. Please install git first."
        exit 1
    fi

    # Check for Claude Code
    if ! command -v claude &> /dev/null; then
        log_warning "Claude Code CLI not found in PATH."
        log_warning "Install it from: https://claude.ai/code"
        echo ""
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        log_success "Claude Code CLI found: $(which claude)"
    fi
}

create_directories() {
    log_info "Creating installation directories..."
    mkdir -p "$SKILLS_CONFIG_DIR"
    mkdir -p "$(dirname "$INSTALL_DIR")"
}

clone_or_update_repo() {
    if [ -d "$INSTALL_DIR/.git" ]; then
        log_info "Existing installation found. Updating..."
        cd "$INSTALL_DIR"
        git fetch origin
        git reset --hard origin/main
        log_success "Updated to latest version"
    else
        log_info "Cloning AI Research Skills repository..."
        if [ -d "$INSTALL_DIR" ]; then
            rm -rf "$INSTALL_DIR"
        fi
        git clone "$REPO_URL" "$INSTALL_DIR"
        log_success "Repository cloned successfully"
    fi
}

setup_claude_config() {
    log_info "Setting up Claude Code configuration..."

    local claude_config="$SKILLS_CONFIG_DIR/settings.json"
    local skills_path="$INSTALL_DIR"

    # Create or update settings.json with skills path
    if [ -f "$claude_config" ]; then
        log_info "Existing Claude config found at $claude_config"
        log_warning "Please manually add the skills path to your configuration:"
        echo ""
        echo -e "  ${YELLOW}Skills directory:${NC} $skills_path"
        echo ""
    else
        log_info "Creating Claude configuration..."
        cat > "$claude_config" << EOF
{
  "skills": {
    "directories": [
      "$skills_path"
    ]
  }
}
EOF
        log_success "Created configuration at $claude_config"
    fi
}

create_claude_md_snippet() {
    log_info "Generating CLAUDE.md snippet for project integration..."

    local snippet_file="$INSTALL_DIR/CLAUDE_MD_SNIPPET.md"

    cat > "$snippet_file" << 'EOF'
# AI Research Skills Integration

To use the AI Research Engineering Skills in your project, add this to your project's CLAUDE.md:

```markdown
## AI Research Skills

This project uses the AI Research Engineering Skills Library.
Skills are available at: ~/.claude/skills/ai-research-skills

### Available Skill Categories:
- Model Architecture (LitGPT, Mamba, NanoGPT, RWKV)
- Tokenization (HuggingFace Tokenizers, SentencePiece)
- Fine-Tuning (Axolotl, LLaMA-Factory, Unsloth, PEFT)
- Mechanistic Interpretability (TransformerLens, SAELens, NNsight, Pyvene)
- Post-Training (TRL, GRPO, OpenRLHF, SimPO)
- Distributed Training (DeepSpeed, FSDP, Accelerate, PyTorch Lightning)
- And more...

To invoke a skill, use: /skill-name (e.g., /axolotl, /grpo-rl-training)
```
EOF

    log_success "Created CLAUDE.md snippet at $snippet_file"
}

print_summary() {
    echo ""
    echo -e "${GREEN}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                 Installation Complete!                        ║${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${BLUE}Installation directory:${NC}"
    echo "  $INSTALL_DIR"
    echo ""
    echo -e "${BLUE}Skills installed:${NC}"

    # Count skills by category
    local total_skills=0
    for category_dir in "$INSTALL_DIR"/[0-9][0-9]-*/; do
        if [ -d "$category_dir" ]; then
            local category_name=$(basename "$category_dir")
            local skill_count=$(find "$category_dir" -maxdepth 1 -type d ! -name "$(basename "$category_dir")" | wc -l | tr -d ' ')
            if [ "$skill_count" -gt 0 ]; then
                echo "  - $category_name: $skill_count skills"
                total_skills=$((total_skills + skill_count))
            fi
        fi
    done
    echo ""
    echo -e "${GREEN}Total: $total_skills skills installed${NC}"
    echo ""
    echo -e "${BLUE}Next steps:${NC}"
    echo "  1. Open a project with Claude Code"
    echo "  2. Reference skills in your CLAUDE.md or invoke them directly"
    echo "  3. Use skills like: /axolotl, /grpo-rl-training, /deepspeed"
    echo ""
    echo -e "${BLUE}To update skills later:${NC}"
    echo "  cd $INSTALL_DIR && git pull"
    echo ""
    echo -e "${BLUE}Documentation:${NC}"
    echo "  $INSTALL_DIR/README.md"
    echo ""
}

# Uninstall function
uninstall() {
    log_info "Uninstalling AI Research Skills..."

    if [ -d "$INSTALL_DIR" ]; then
        rm -rf "$INSTALL_DIR"
        log_success "Removed $INSTALL_DIR"
    else
        log_warning "Installation directory not found"
    fi

    log_success "Uninstallation complete"
    echo ""
    echo "Note: You may want to manually remove the skills reference from:"
    echo "  $SKILLS_CONFIG_DIR/settings.json"
}

# Main execution
main() {
    print_banner

    # Check for uninstall flag
    if [ "$1" = "--uninstall" ] || [ "$1" = "-u" ]; then
        uninstall
        exit 0
    fi

    # Check for help flag
    if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  --help, -h       Show this help message"
        echo "  --uninstall, -u  Uninstall the skills library"
        echo ""
        echo "Environment variables:"
        echo "  REPO_URL         Custom repository URL"
        echo "  INSTALL_DIR      Custom installation directory"
        echo ""
        exit 0
    fi

    check_dependencies
    create_directories
    clone_or_update_repo
    setup_claude_config
    create_claude_md_snippet
    print_summary
}

main "$@"
