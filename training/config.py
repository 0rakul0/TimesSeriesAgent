"""
config.py — Configurações globais do framework v5
"""

import torch

# ========================
# Dados
# ========================
DATA_PATH = "../data/dados_combinados.csv"
TARGET_COL = "Close_PETR4.SA"
SEQ_LEN = 30

# ========================
# Treino
# ========================
BATCH_SIZE = 64
LR = 5e-4
EPOCHS = 40
PATIENCE = 6    # early stopping
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========================
# Saídas
# ========================
MODEL_DIR = "./models"
OUTPUT_DIR = "./outputs"

# Cria pastas automaticamente
import os
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
