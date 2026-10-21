# %% [markdown]
# # Churn com impacto em R$ — Telco (IBM)
# Objetivo: prever quais clientes vão cancelar e descobrir qual estratégia
# de retenção gera mais economia.

# %% 1) IMPORTS E CARGA DOS DADOS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, roc_auc_score,
                             recall_score, precision_score, confusion_matrix)

sns.set_theme(style="whitegrid")

df = pd.read_excel("data/Telco_customer_churn.xlsx")
print(df.shape)  # esperado: (7043, 33)
print(df.columns.tolist())  # se der KeyError adiante, confira os nomes aqui
df.head()

# %% 2) LIMPEZA E TRADUÇÃO
# "Total Charges" vem como texto, com alguns espaços vazios (clientes novos)
df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
print("Nulos em Total Charges:", df["Total Charges"].isna().sum())
df["Total Charges"] = df["Total Charges"].fillna(0)

# Guarda os motivos de saída ANTES de apagar (vira gráfico, mas NÃO entra no modelo)
traducao_motivos = {
    "Attitude of support person": "Atitude do atendente de suporte",
    "Attitude of service provider": "Atitude da operadora",
    "Competitor offered higher download speeds": "Concorrente ofereceu mais velocidade",
    "Competitor offered more data": "Concorrente ofereceu mais dados",
    "Competitor made better offer": "Concorrente fez oferta melhor",
    "Competitor had better devices": "Concorrente tinha aparelhos melhores",
    "Don't know": "Não sabe",
    "Deceased": "Falecimento",
    "Extra data charges": "Cobranças extras de dados",
    "Lack of affordable download/upload speed": "Velocidade acessível insuficiente",
    "Lack of self-service on Website": "Falta de autoatendimento no site",
    "Limited range of services": "Poucos serviços disponíveis",
    "Long distance charges": "Tarifas de longa distância",
    "Moved": "Mudou de endereço",
    "Network reliability": "Instabilidade da rede",
    "Poor expertise of online support": "Suporte online fraco",
    "Poor expertise of phone support": "Suporte por telefone fraco",
    "Price too high": "Preço muito alto",
    "Product dissatisfaction": "Insatisfação com o produto",
    "Service dissatisfaction": "Insatisfação com o serviço",
}
motivos = (df.loc[df["Churn Value"] == 1, "Churn Reason"]
             .value_counts().head(10)
             .rename(index=traducao_motivos))

# Remove colunas inúteis e as que VAZAM a resposta (data leakage)
remover = [
    "CustomerID", "Count", "Country", "State", "City", "Zip Code",
    "Lat Long", "Latitude", "Longitude",
    "Churn Label", "Churn Score", "CLTV", "Churn Reason",
]
df = df.drop(columns=remover)

# Traduz os nomes das colunas
df = df.rename(columns={
    "Gender": "Gênero",
    "Senior Citizen": "Idoso",
    "Partner": "Tem parceiro(a)",
    "Dependents": "Tem dependentes",
    "Tenure Months": "Tempo de casa (meses)",
    "Phone Service": "Telefone",
    "Multiple Lines": "Várias linhas",
    "Internet Service": "Tipo de internet",
    "Online Security": "Segurança online",
    "Online Backup": "Backup online",
    "Device Protection": "Proteção de aparelho",
    "Tech Support": "Suporte técnico",
    "Streaming TV": "Streaming de TV",
    "Streaming Movies": "Streaming de filmes",
    "Contract": "Contrato",
    "Paperless Billing": "Fatura digital",
    "Payment Method": "Forma de pagamento",
    "Monthly Charges": "Cobrança mensal",
    "Total Charges": "Cobrança total",
    "Churn Value": "Cancelou",
})

# Traduz os valores das categorias
valores = {
    "Yes": "Sim", "No": "Não",
    "No phone service": "Sem telefone",
    "No internet service": "Sem internet",
    "Male": "Masculino", "Female": "Feminino",
    "Fiber optic": "Fibra óptica",
    "Month-to-month": "Mensal", "One year": "1 ano", "Two year": "2 anos",
    "Mailed check": "Cheque pelo correio",
    "Electronic check": "Cheque eletrônico",
    "Bank transfer (automatic)": "Débito automático",
    "Credit card (automatic)": "Cartão (automático)",
}
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].replace(valores)

# "Não" em tipo de internet significa que o cliente não tem internet
df["Tipo de internet"] = df["Tipo de internet"].replace({"Não": "Sem internet"})

print("Taxa de cancelamento:", round(df["Cancelou"].mean() * 100, 1), "%")
df.info()

# %% 3) ANÁLISE EXPLORATÓRIA
df_graf = df.copy()
df_graf["Situação"] = df_graf["Cancelou"].map({0: "Ficou", 1: "Saiu"})

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# Cancelamento por tipo de contrato
taxa = df_graf.groupby("Contrato")["Cancelou"].mean().sort_values() * 100
taxa.plot(kind="barh", ax=axes[0, 0], color="#d62728")
axes[0, 0].set_title("Taxa de cancelamento por tipo de contrato")
axes[0, 0].set_xlabel("Taxa de cancelamento (%)")
axes[0, 0].set_ylabel("")

# Cancelamento por tipo de internet
taxa = df_graf.groupby("Tipo de internet")["Cancelou"].mean().sort_values() * 100
taxa.plot(kind="barh", ax=axes[0, 1], color="#d62728")
axes[0, 1].set_title("Taxa de cancelamento por tipo de internet")
axes[0, 1].set_xlabel("Taxa de cancelamento (%)")
axes[0, 1].set_ylabel("")

# Tempo de casa
sns.histplot(data=df_graf, x="Tempo de casa (meses)", hue="Situação", bins=30,
             multiple="stack", ax=axes[1, 0])
axes[1, 0].set_title("Tempo de casa dos clientes")
axes[1, 0].set_xlabel("Tempo de casa (meses)")
axes[1, 0].set_ylabel("Número de clientes")

# Cobrança mensal
sns.boxplot(data=df_graf, x="Situação", y="Cobrança mensal", ax=axes[1, 1])
axes[1, 1].set_title("Cobrança mensal: quem ficou x quem saiu")
axes[1, 1].set_xlabel("")
axes[1, 1].set_ylabel("Cobrança mensal (US$)")

plt.tight_layout()
plt.savefig("images/eda.png", dpi=150)
plt.show()

# %% 3.1) POR QUE OS CLIENTES SAEM? (ótimo para o LinkedIn)
plt.figure(figsize=(11, 6))
motivos.sort_values().plot(kind="barh", color="#1f77b4")
plt.title("Top 10 motivos de cancelamento")
plt.xlabel("Número de clientes")
plt.ylabel("")
plt.tight_layout()
plt.savefig("images/motivos_cancelamento.png", dpi=150)
plt.show()

# %% 4) PREPARAÇÃO E DIVISÃO TREINO/TESTE
y = df["Cancelou"]
X = pd.get_dummies(df.drop(columns=["Cancelou"]), drop_first=True)

# stratify=y mantém a mesma proporção de cancelamentos em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)
print("Treino:", X_train.shape, "| Teste:", X_test.shape)

# %% 5) MODELOS
# Regressão logística (precisa de escala) e Random Forest
logit = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, class_weight="balanced"),
)
rf = RandomForestClassifier(
    n_estimators=300, min_samples_leaf=5,
    class_weight="balanced", random_state=42, n_jobs=-1,
)

modelos = {"Regressão Logística": logit, "Random Forest": rf}
probas = {}

for nome, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    p = modelo.predict_proba(X_test)[:, 1]
    probas[nome] = p
    pred = (p >= 0.5).astype(int)
    print(f"\n===== {nome} =====")
    print("AUC:", round(roc_auc_score(y_test, p), 3))
    print(classification_report(y_test, pred, target_names=["Ficou", "Saiu"]))

# %% 6) IMPACTO EM R$ (o diferencial do projeto)
# ATENÇÃO: os valores abaixo são PREMISSAS suas, não dados reais.
# Diga isso no README.
CUSTO_RETENCAO = 50    # R$ gastos para tentar reter 1 cliente (desconto, ligação...)
PERDA_CLIENTE = 600    # R$ perdidos quando 1 cliente cancela
TAXA_SUCESSO = 0.30    # % dos clientes abordados que realmente ficam

escala = len(df) / len(y_test)  # extrapola do teste para a base inteira

def economia(y_real, p, corte):
    abordados = p >= corte
    verdadeiros = abordados & (y_real.values == 1)
    ganho = verdadeiros.sum() * TAXA_SUCESSO * PERDA_CLIENTE
    custo = abordados.sum() * CUSTO_RETENCAO
    return (ganho - custo) * escala

cortes = np.arange(0.05, 0.96, 0.05)
resultado = pd.DataFrame({
    nome: [economia(y_test, p, c) for c in cortes]
    for nome, p in probas.items()
}, index=cortes.round(2))

melhor_nome = resultado.max().idxmax()
melhor_corte = resultado[melhor_nome].idxmax()
melhor_valor = resultado[melhor_nome].max()

# Comparação com "abordar todos os clientes"
todos = ((y_test.sum() * TAXA_SUCESSO * PERDA_CLIENTE)
         - len(y_test) * CUSTO_RETENCAO) * escala

print(f"Melhor modelo: {melhor_nome} | corte: {melhor_corte}")
print(f"Economia estimada: R$ {melhor_valor:,.0f} por trimestre")
print(f"Se abordar todos os clientes: R$ {todos:,.0f} por trimestre")

plt.figure(figsize=(10, 6))
for nome in resultado.columns:
    plt.plot(resultado.index, resultado[nome], marker="o", label=nome)
plt.axhline(0, color="gray", linestyle="--")
plt.scatter([melhor_corte], [melhor_valor], s=200, color="gold",
            edgecolor="black", zorder=5, label="Melhor ponto")
plt.title("Economia estimada (R$ por trimestre) conforme o corte de probabilidade")
plt.xlabel("Corte de probabilidade para abordar o cliente")
plt.ylabel("Economia (R$)")
plt.legend()
plt.tight_layout()
plt.savefig("images/economia_por_corte.png", dpi=150)
plt.show()

# %% 7) O QUE MAIS PESA NO CANCELAMENTO?
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values().tail(10)
plt.figure(figsize=(10, 6))
imp.plot(kind="barh", color="#2ca02c")
plt.title("Top 10 variáveis mais importantes (Random Forest)")
plt.xlabel("Importância")
plt.tight_layout()
plt.savefig("images/importancia.png", dpi=150)
plt.show()

# %% 8) ACERTOS E ERROS DO MELHOR MODELO
p = probas[melhor_nome]
pred = (p >= melhor_corte).astype(int)
cm = confusion_matrix(y_test, pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Previsto: ficou", "Previsto: saiu"],
            yticklabels=["Real: ficou", "Real: saiu"])
plt.title(f"Matriz de confusão — {melhor_nome}")
plt.tight_layout()
plt.savefig("images/matriz_confusao.png", dpi=150)
plt.show()

print("Recall (dos que saíram, quantos o modelo pegou):",
      round(recall_score(y_test, pred), 3))
print("Precisão (dos que o modelo apontou, quantos saíram de fato):",
      round(precision_score(y_test, pred), 3))