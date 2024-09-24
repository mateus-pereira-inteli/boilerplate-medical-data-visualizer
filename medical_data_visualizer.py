import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def draw_cat_plot():
    # Carregar os dados
    df = pd.read_csv('medical_examination.csv')

    # Adicionar a coluna 'overweight': 1 se IMC > 25, caso contrário 0
    df['overweight'] = (df['weight'] / (df['height'] / 100) ** 2).apply(lambda x: 1 if x > 25 else 0)

    # Normalizar os valores de colesterol e glicose para 0 se for 1, e 1 se for maior que 1
    df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
    df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)

    # Reorganizar os dados para criar a visualização
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke'])

    # Criar o gráfico catplot e ajustar o rótulo do eixo y para 'total'
    fig = sns.catplot(x='variable', hue='value', col='cardio', data=df_cat, kind='count').fig
    fig.axes[0].set_ylabel('total')  # Ajustar rótulo do eixo y

    # Salvar a figura
    fig.savefig('catplot.png')
    return fig

def draw_heat_map():
    # Carregar os dados
    df = pd.read_csv('medical_examination.csv')

    # **Adicionar a coluna 'overweight' antes de filtrar os dados**
    df['overweight'] = (df['weight'] / (df['height'] / 100) ** 2).apply(lambda x: 1 if x > 25 else 0)

    # **Normalizar os valores de colesterol e glicose antes de filtrar os dados**
    df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
    df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)

    # Filtrar os dados para remover linhas inválidas
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # Calcular a correlação entre as variáveis
    corr = df_heat.corr()

    # Gerar uma máscara para a parte superior da matriz de correlação
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Configurar a figura do matplotlib
    fig, ax = plt.subplots(figsize=(12, 12))

    # Desenhar o mapa de calor
    sns.heatmap(
        corr,
        annot=True,
        fmt='.1f',
        mask=mask,
        square=True,
        center=0,
        vmin=-0.16,
        vmax=0.32,
        cbar_kws={'shrink': 0.5},
        cmap='coolwarm'
    )

    # Ajustar os rótulos
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=45)

    # Salvar a figura
    fig.savefig('heatmap.png')
    return fig
