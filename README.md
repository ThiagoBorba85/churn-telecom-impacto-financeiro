# Previsão de Cancelamento de Clientes (Churn) em Telecom

Projeto de ciência de dados feito por mim para prever quais clientes de uma telecom vão cancelar e calcular quanto dinheiro a empresa pode economizar com isso.

![Economia por corte de probabilidade](Images/economia_por_corte.png)

## O problema

Quando um cliente cancela, a empresa perde dinheiro. Mas tentar reter todos os clientes também custa caro. Então a pergunta do projeto é: **quais clientes vale a pena abordar?**

## Os dados

- Dataset **Telco Customer Churn (IBM)**, baixado do **Kaggle**.
- 7.043 clientes e 33 colunas de uma telecom **fictícia** da Califórnia (EUA).
- Link: https://www.kaggle.com/search?q=Telco+Customer+Churn+IBM
- Os dados são da IBM e foram usados só para estudo e portfólio.

## O que eu fiz

1. Limpei os dados e traduzi os nomes para português.
2. Removi as colunas `Churn Score`, `CLTV` e `Churn Reason`, porque elas já entregam a resposta ao modelo (vazamento de dados).
3. Fiz a análise exploratória para entender quem cancela.
4. Separei os dados em treino (75%) e teste (25%).
5. Treinei e comparei dois modelos: Regressão Logística e Random Forest.
6. Calculei quanto dinheiro cada estratégia economiza.

## O que descobri

![Análise exploratória](Images/eda.png)

- Contrato **mensal**: cerca de 42% cancelam. Contrato de **2 anos**: cerca de 3%.
- Internet por **fibra óptica**: cerca de 42% cancelam. DSL: cerca de 19%.
- A maioria dos cancelamentos acontece nos **primeiros meses** do cliente.
- Quem cancelou pagava mais por mês.

![Motivos de cancelamento](Images/motivos_cancelamento.png)

- Entre os 10 principais motivos, **atendimento ruim** e **ofertas de concorrentes** aparecem bem acima de preço.

![Variáveis mais importantes](Images/importancia.png)

## Resultado em dinheiro

Usei estas **suposições minhas** (não são dados reais da empresa):

| Suposição | Valor |
|---|---|
| Custo para abordar 1 cliente | R$ 50 |
| Perda quando 1 cliente cancela | R$ 600 |
| Chance de conseguir reter o cliente | 30% |

**Melhor resultado:** Random Forest com corte de 0,45.

- Economia estimada: **R$ 130.901 por trimestre**
- Se a empresa abordasse **todos** os clientes: **prejuízo de R$ 15.958 por trimestre**
- O modelo encontra **79%** dos clientes que cancelam (recall de 0,794)
- Dos clientes que o modelo aponta, **55%** realmente cancelariam (precisão de 0,545)

Por que uma precisão de 55% é boa? Porque abordar custa R$ 50 e o ganho esperado por cliente que cancelaria é de R$ 180 (30% de R$ 600). A conta já fecha com precisão acima de cerca de 28%.

## O que eu recomendaria à empresa

1. Incentivar a troca do contrato mensal para o anual, principalmente nos primeiros meses.
2. Melhorar a qualidade do atendimento.
3. Investigar a velocidade e a estabilidade da fibra óptica.

## Limitações

- Os valores em R$ são suposições minhas.
- A base é fictícia e de um único trimestre.
- A importância das variáveis mostra o que pesa no modelo, mas não prova causa e efeito.

## Como rodar

1. Instale as bibliotecas: `pip install -r requirements.txt`
2. Baixe a base no Kaggle (link acima) e coloque na pasta `data/`
3. Rode o arquivo `churn_projeto.py` bloco a bloco no VS Code

## Ferramentas

Python, pandas, NumPy, scikit-learn, Matplotlib e Seaborn.

## Autor

Thiago Rosa de Borba — estudante de Ciência de Dados (Uninter)
LinkedIn: https://www.linkedin.com/in/thiagoborba85

git config --global user.name Thiago Borba
git config --global user.email thiagodr88@gmail.com