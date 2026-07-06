import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Configura para execução sem interface gráfica
import matplotlib.pyplot as plt
import seaborn as sns

# Definindo a paleta de cores
sns.set_theme(style='whitegrid', palette='colorblind', font='DejaVu Sans')

# Carrega os dados
df = pd.read_csv('data/water_potability.csv')

# 1. Distribuição de Classes
fig, ax = plt.subplots(figsize=(8, 6))
counts = df['Potability'].value_counts()
pcts = df['Potability'].value_counts(normalize=True) * 100

bars = sns.barplot(x=['Não Potável (Classe 0)', 'Potável (Classe 1)'], y=counts.values, ax=ax, palette='colorblind')

# Adiciona os rótulos de valores e porcentagens no topo das barras
for i, bar in enumerate(ax.patches):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 30,
            f'{int(height)} ({pcts.iloc[i]:.1f}%)',
            ha="center", va="bottom", fontweight='bold', fontsize=11)

ax.set_title('Desbalanceamento de Classes: 61% das Amostras são Não-Potáveis\n(Exigindo Balanceamento com SMOTE no Pipeline)', fontsize=12, fontweight='bold', pad=15)
ax.set_xlabel('Classe de Potabilidade')
ax.set_ylabel('Número de Amostras')
ax.set_ylim(0, 2300)
sns.despine()
plt.tight_layout(pad=1.5)
fig.savefig('distribuicao_classes.png', dpi=150, bbox_inches='tight')
plt.close()

# 2. Heatmap de Correlação
fig, ax = plt.subplots(figsize=(10, 8))
corr = df.corr()

# Máscara para ocultar a metade superior espelhada (deixa o gráfico mais limpo)
mask = np.triu(np.ones_like(corr, dtype=bool))

sns.heatmap(corr, mask=mask, annot=True, fmt='.3f', cmap='coolwarm', center=0,
            square=True, linewidths=0.5, ax=ax, vmin=-0.1, vmax=0.1, cbar_kws={"shrink": .8})

ax.set_title('Correlação Quase Nula: Variáveis Apresentam Relações Lineares Desprezíveis (r < 0.05)\n(Comprova a Necessidade de Modelos Não-Lineares como Random Forest e XGBoost)', fontsize=11, fontweight='bold', pad=15)
plt.tight_layout(pad=1.5)
fig.savefig('heatmap_correlacao.png', dpi=150, bbox_inches='tight')
plt.close()

# 3. Distribuição do pH com Limites da OMS
fig, ax = plt.subplots(figsize=(10, 6))
ph_data = df['ph'].dropna()

sns.histplot(ph_data, kde=True, ax=ax, color='#55a868', bins=40, edgecolor='white', alpha=0.6)

# Linhas de limites recomendados pela OMS
ax.axvline(6.5, color='red', linestyle='--', linewidth=1.5, label='Limite Mínimo OMS (6.5)')
ax.axvline(8.5, color='red', linestyle='--', linewidth=1.5, label='Limite Máximo OMS (8.5)')
ax.axvspan(6.5, 8.5, color='green', alpha=0.15, label='Faixa Recomendada OMS (6.5 - 8.5)')

ax.set_title('pH Fora dos Padrões: Amostras de Água Excedem os Limites Saudáveis (6.5 - 8.5) da OMS\n(Distribuição Aproximadamente Normal com Média de {:.2f})'.format(ph_data.mean()), fontsize=12, fontweight='bold', pad=15)
ax.set_xlabel('Valor de pH')
ax.set_ylabel('Frequência de Amostras')
ax.legend(loc='upper right', frameon=True)
sns.despine()
plt.tight_layout(pad=1.5)
fig.savefig('distribuicao_ph.png', dpi=150, bbox_inches='tight')
plt.close()